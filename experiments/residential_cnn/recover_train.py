"""Train the original Multi-Scale CNN and preserve a resumable experiment on persistent storage."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import MultiScaleCNN

SCHEMA_VERSION = 1


def as_tensor(array):
    # This host has NumPy 2 and PyTorch compiled against the NumPy 1 ABI.
    value = np.ascontiguousarray(array, dtype=np.float32)
    return torch.frombuffer(value, dtype=torch.float32).reshape(value.shape)


def as_array(tensor):
    value = tensor.detach().cpu()
    return np.asarray(value.tolist(), dtype=str(value.dtype).removeprefix('torch.'))


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_digest(state):
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        hasher.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(as_array(value).tobytes())
    return hasher.hexdigest()


def data_digest(data):
    hasher = hashlib.sha256()
    for key in ("X", "Y", "split", "recording", "byte_order"):
        value = np.ascontiguousarray(data[key])
        hasher.update(key.encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(str(value.shape).encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def atomic_save(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    atomic_save(path, lambda temporary: temporary.write_text(json.dumps(value, indent=2) + "\n"))


def save_npz(path, **items):
    def write(temporary):
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **items)
    atomic_save(path, write)


class Packets(Dataset):
    def __init__(self, x, y, augment=False):
        self.x, self.y, self.augment = as_tensor(x), as_tensor(y), augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index].view(2, 128)
        if self.augment:
            x = (x + torch.randn_like(x) * 0.05) * (torch.rand(128) > 0.05)
        return x, self.y[index]


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in ("X", "Y", "split", "recording", "byte_order")}
    x, y, split, recording = [values[key] for key in ("X", "Y", "split", "recording")]
    if x.shape != (len(y), 256) or y.shape != (len(x), 3):
        raise ValueError("Expected 256 CSI features and three position labels")
    if len(split) != len(x) or len(recording) != len(x) or set(split) != {"train", "val", "test"}:
        raise ValueError("Expected one partition and recording label per packet, with all partitions nonempty")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("All input values must be finite")
    for name in np.unique(recording):
        if len(np.unique(split[recording == name])) != 1:
            raise ValueError(f"Recording spans multiple partitions: {name}")
    return values


def random_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "numpy_name": numpy_state[0],
        "numpy_keys": numpy_state[1].tolist(), "numpy_position": numpy_state[2],
        "numpy_gaussian": numpy_state[3], "numpy_cached": numpy_state[4],
        "torch": torch.get_rng_state(), "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state, generator):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"], np.asarray(state["numpy_keys"], dtype=np.uint32),
                         state["numpy_position"], state["numpy_gaussian"], state["numpy_cached"]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.inference_mode()
def predict(model, loader, device, mean, std):
    model.eval()
    predictions = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            value = model(x.to(device))
        predictions.append(as_array(value.float()) * std + mean)
    return np.concatenate(predictions)


def evaluate(predictions, labels, recording):
    errors = np.linalg.norm(predictions - labels, axis=1)
    per_recording = {
        str(name): float(errors[recording == name].mean()) for name in np.unique(recording)
    }
    return {
        "packets": len(labels), "recordings": len(per_recording),
        "mean_distance_m": float(errors.mean()), "median_distance_m": float(np.median(errors)),
        "p90_distance_m": float(np.percentile(errors, 90)),
        "recording_balanced_mean_distance_m": float(np.mean(list(per_recording.values()))),
        "per_axis_mae_m": dict(zip("xyz", np.abs(predictions - labels).mean(axis=0).tolist())),
        "per_recording_mean_distance_m": per_recording,
    }


def run(args):
    if args.epochs < 1 or args.batch_size < 2 or (args.max_epochs_this_call is not None and args.max_epochs_this_call < 1):
        raise ValueError("Epoch counts must be positive and batch size must be at least two")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    data = load_data(args.data)
    split, recording = data["split"], data["recording"]
    x, y, order = data["X"], data["Y"], str(data["byte_order"].item())
    train = split == "train"
    x_mean, x_std = x[train].mean(axis=0), x[train].std(axis=0) + 1e-8
    y_mean, y_std = y[train].mean(axis=0), y[train].std(axis=0) + 1e-8
    x_scaled = ((x - x_mean) / x_std).astype(np.float32)
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    initial = torch.load(args.init, map_location="cpu", weights_only=True) if args.init else None
    normalization = {
        key: as_tensor(value.copy())
        for key, value in dict(X_mean=x_mean, X_std=x_std, Y_mean=y_mean, Y_std=y_std).items()
    }
    config = {
        "schema": SCHEMA_VERSION, "model": "MultiScaleCNN", "data_sha256": data_digest(data),
        "initial_model_state_sha256": model_digest(initial["model_state"]) if initial else None,
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "phase": "finetune" if args.init else "pretrain", "byte_order": order,
        "learning_rate": 5e-4 if args.init else 1e-3,
        "weight_decay": 1e-3 if args.init else 1e-4,
        "schedule": "cosine" if args.init else "onecycle",
        "loss": "Huber, delta=1 in normalized target space",
        "augmentation": {"gaussian_std": 0.05, "mask_fraction": 0.05},
        "model_selection": "validation mean Euclidean distance in meters",
        "run_schema": SCHEMA_VERSION, "model_source_sha256": sha256(Path(__file__).with_name("models.py")),
        "runner_source_sha256": sha256(Path(__file__)),
        "device": str(device), "threads": args.threads,
        "versions": {"python": platform.python_version(), "numpy": str(np.__version__),
                     "torch": str(torch.__version__), "cuda": str(torch.version.cuda)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "last.pt"
    if any(args.out.iterdir()) and not args.resume:
        raise ValueError("Output directory contains files. Use --resume or a new directory")
    if args.resume and not checkpoint_path.exists():
        raise ValueError("No last.pt exists in the requested resume directory")

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for partition in ("train", "val", "test"):
        mask = split == partition
        dataset = Packets(x_scaled[mask], y_scaled[mask], augment=partition == "train")
        loaders[partition] = DataLoader(dataset, batch_size=args.batch_size,
                                       shuffle=partition == "train", num_workers=0,
                                       generator=generator if partition == "train" else None)
    model = MultiScaleCNN().to(device)
    if args.init:
        if initial["config"]["model"] != "MultiScaleCNN" or initial["config"]["byte_order"] != order:
            raise ValueError("Initial checkpoint model or CSI feature order differs from this dataset")
        model.load_state_dict(initial["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    if args.init:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-3, epochs=args.epochs, steps_per_epoch=len(loaders["train"])
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, start_epoch, best, best_epoch = [], 0, float("inf"), 0
    best_state = None
    if args.resume:
        previous = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if previous["config"] != config:
            raise ValueError("Run settings, source files, environment, or input data differ from the saved run")
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        scheduler.load_state_dict(previous["scheduler_state"])
        scaler.load_state_dict(previous["scaler_state"])
        history, start_epoch = previous["history"], previous["epoch"]
        best, best_epoch = previous["best_validation_error_m"], previous["best_epoch"]
        restore_random_state(previous["random_state"], generator)
        best_state = previous["best_model_state"]
        atomic_save(args.out / "best.pt", lambda path: torch.save({
            "schema": SCHEMA_VERSION, "config": config, "epoch": best_epoch,
            "model_state": best_state, "normalization": normalization,
        }, path))
    else:
        save_json(args.out / "config.json", config)
        save_json(args.out / "splits.json", {
            name: {"recordings": sorted(set(recording[split == name].tolist())),
                   "packets": int(np.sum(split == name))} for name in ("train", "val", "test")
        })
        save_npz(args.out / "normalization.npz", **{key: as_array(value) for key, value in normalization.items()})
        provenance = args.data.with_suffix(".provenance.json")
        if provenance.exists():
            save_json(args.out / "input_provenance.json", json.loads(provenance.read_text()))

    final_epoch = args.epochs
    if args.max_epochs_this_call:
        final_epoch = min(args.epochs, start_epoch + args.max_epochs_this_call)
    loss_fn = nn.HuberLoss(delta=1.0)
    for epoch in range(start_epoch, final_epoch):
        model.train()
        total = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not args.init:
                scheduler.step()
            total += float(loss.item()) * len(xb)
        if args.init:
            scheduler.step()
        val = split == "val"
        predictions = predict(model, loaders["val"], device, y_mean, y_std)
        error = float(np.linalg.norm(predictions - y[val], axis=1).mean())
        if not np.isfinite(error):
            raise RuntimeError("Validation error became nonfinite")
        history.append({"epoch": epoch + 1, "training_loss": total / int(train.sum()),
                        "validation_mean_distance_m": error})
        common = {"schema": SCHEMA_VERSION, "config": config, "epoch": epoch + 1,
                  "model_state": model.state_dict(), "normalization": normalization}
        if error < best:
            best, best_epoch = error, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            atomic_save(args.out / "best.pt", lambda path: torch.save(common, path))
        state = dict(common, optimizer_state=optimizer.state_dict(),
                     scheduler_state=scheduler.state_dict(), scaler_state=scaler.state_dict(),
                     random_state=random_state(generator), history=history,
                     best_validation_error_m=best, best_epoch=best_epoch,
                     best_model_state=best_state)
        atomic_save(checkpoint_path, lambda path: torch.save(state, path))
        save_json(args.out / "history.json", history)
        print(f"Epoch {epoch+1}/{args.epochs}: validation {error:.4f} m, best {best:.4f} m; checkpoint saved",
              flush=True)

    if final_epoch < args.epochs:
        print("Paused at an epoch boundary. Continue with the same command plus --resume.", flush=True)
        return
    best_model = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best_model["model_state"])
    results = {"config": config, "best_epoch": best_epoch,
               "best_validation_error_m": best, "history": history}
    for partition in ("val", "test"):
        mask = split == partition
        predictions = predict(model, loaders[partition], device, y_mean, y_std)
        results[partition] = evaluate(predictions, y[mask], recording[mask])
        save_npz(args.out / f"{partition}_predictions.npz",
                 predictions=predictions, labels=y[mask], recording=recording[mask])
        with (args.out / f"{partition}_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["recording", "pred_x_m", "pred_y_m", "pred_z_m",
                             "true_x_m", "true_y_m", "true_z_m", "distance_m"])
            for name, pred, label in zip(recording[mask], predictions, y[mask]):
                writer.writerow([name, *pred.tolist(), *label.tolist(), float(np.linalg.norm(pred - label))])
    save_json(args.out / "metrics.json", results)
    save_json(args.out / "artifacts_sha256.json", {
        file.name: sha256(file) for file in sorted(args.out.iterdir())
        if file.is_file() and file.name != "artifacts_sha256.json"
    })
    print(f"Finished. Held-out test mean distance: {results['test']['mean_distance_m']:.4f} m", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Use a Google Drive folder in Colab")
    parser.add_argument("--init", type=Path, help="best.pt from the source training run")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-call", type=int, help="Pause cleanly after this many more epochs")
    run(parser.parse_args())
