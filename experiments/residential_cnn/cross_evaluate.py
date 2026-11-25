"""Score the two reconstructed CNN checkpoints on the same frozen test recordings."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from models import MultiScaleCNN
from recover_train import as_tensor, as_array, evaluate, load_data


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["model"] != "MultiScaleCNN":
        raise ValueError(f"Unexpected checkpoint architecture: {path}")
    return checkpoint


@torch.inference_mode()
def score(checkpoint, data, device, norm=None, use_amp=True):
    norm = norm or checkpoint["normalization"]
    if checkpoint["config"]["byte_order"] != str(data["byte_order"].item()):
        raise ValueError("Checkpoint and dataset use different byte conventions")
    partition = data["split"] == "test"
    features = data["X"][partition]
    labels = data["Y"][partition]
    recording = data["recording"][partition]
    x_mean, x_std, y_mean, y_std = (as_array(norm[k]) for k in ("X_mean", "X_std", "Y_mean", "Y_std"))
    standardized = (features - x_mean) / x_std
    scaled = as_tensor(standardized).reshape(-1, 2, 128)
    model = MultiScaleCNN().eval().to(device)
    model.load_state_dict(checkpoint["model_state"])
    batches = []
    for batch in scaled.split(512):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and use_amp):
            output = model(batch.to(device))
        batches.append(as_array(output.float()) * y_std + y_mean)
    predictions = np.concatenate(batches)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite predictions")
    metrics = evaluate(predictions, labels, recording)
    metrics["max_absolute_standardized_feature"] = float(np.abs(standardized).max())
    metrics["mixed_precision"] = device.type == "cuda" and use_amp
    horizontal = np.linalg.norm(predictions[:, :2] - labels[:, :2], axis=1)
    metrics["mean_horizontal_distance_m"] = float(horizontal.mean())
    metrics["per_recording_mean_horizontal_distance_m"] = {
        str(name): float(horizontal[recording == name].mean()) for name in np.unique(recording)
    }
    return metrics, predictions, labels, recording


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--residential-data", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--finetuned", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, res = load_data(a.source_data), load_data(a.residential_data)
    before, after = read_checkpoint(a.pretrained), read_checkpoint(a.finetuned)
    measures = {}
    for name, model, data, norm, amp in (
        ("pretrained_on_HALOC", before, src, None, False),
        ("finetuned_on_HALOC", after, src, None, False),
        ("finetuned_with_HALOC_calibration", after, src, before["normalization"], False),
        ("pretrained_on_residential", before, res, None, False),
        ("pretrained_with_residential_calibration", before, res, after["normalization"], False),
        ("finetuned_on_residential", after, res, None, False),
    ):
        measures[name], pred, true, files = score(model, data, device, norm, use_amp=amp)
        np.savez_compressed(a.out / (name + "_predictions.npz"), predictions=pred, labels=true, recording=files)
        print(f"{name}: {measures[name]['mean_distance_m']:.4f} m", flush=True)
    report = {
        "checkpoint_epochs": {"pretrained": before["epoch"], "finetuned": after["epoch"]},
        "checkpoint_configs": {"pretrained": before["config"], "finetuned": after["config"]},
        "test_metrics": measures,
        "normalization": (
            "Each pipeline uses the training statistics saved with its checkpoint. "
            "The calibration controls keep one checkpoint's weights and use the other "
            "domain's training statistics. All cross-evaluations use full precision."
        ),
    }
    (a.out / "cross_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
