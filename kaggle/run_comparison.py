"""Reproduce the four ThroughWi Kaggle conditions on a CUDA machine."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DATA_COMMIT = "6bff28e8aec35c85365b95ffc1d5d0631245b973"
CODE_COMMIT = "af54c0cdd1e2be3a0d0c0f524ebef15db50bf87e"
SOURCE_SHA256 = "a77dda823d02c2d900a5908972b1752be1d1dc4866a48cb032ded29273ea52c7"


def checkout_paths(commit, paths, destination):
    archived = subprocess.check_output(
        ["git", "-C", str(REPO), "archive", "--format=zip", commit, *paths]
    )
    with zipfile.ZipFile(io.BytesIO(archived)) as archive:
        archive.extractall(destination)


def append_training(dataset, scenarios, csi_features, excluded=()):
    features, labels, names = [], [], []
    for path, name, xyz in scenarios:
        opener = gzip.open if path.suffix == ".gz" else open
        before = len(names)
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                parts = row["csi_raw"].split()
                if float(row["csi_len"]) != 256 or len(parts) != 256:
                    continue
                values = np.asarray(parts, dtype=np.float32)
                if not np.isfinite(values).all():
                    continue
                feature = np.asarray(csi_features(values, "iq"), dtype=dataset["X"].dtype)
                if feature.tobytes() in excluded:
                    continue
                features.append(feature)
                labels.append(xyz)
                names.append(name)
        if len(names) == before:
            raise ValueError(f"No usable training packets in {path}")
    merged = dict(dataset)
    merged["recording"] = dataset["recording"].astype("U256")
    additions = {"X": features, "Y": labels, "recording": names, "split": ["train"] * len(names)}
    for key, values in additions.items():
        merged[key] = np.concatenate([merged[key], np.asarray(values, dtype=merged[key].dtype)])
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU before running the comparison.")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = REPO / "kaggle" / "haloc-best.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("HALOC checkpoint checksum mismatch.")

    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    checkout_paths(DATA_COMMIT, ["data"], inputs)
    checkout_paths(CODE_COMMIT, ["experiments/residential_cnn"], inputs)
    data = inputs / "data"
    code = inputs / "experiments" / "residential_cnn"
    sys.path.insert(0, str(code))
    from prepare_captures import csi_features
    from cross_evaluate import read_checkpoint, score
    from recover_train import load_data

    base = out / "base"
    subprocess.run(
        [sys.executable, str(code / "prepare_captures.py"), "--data", str(data),
         "--out", str(base), "--byte-order", "iq"], check=True
    )
    original_path = base / "captures.npz"
    with np.load(original_path, allow_pickle=False) as dataset:
        original = dict(dataset)
    with (data / "recordings.csv").open(newline="", encoding="utf-8-sig") as handle:
        train = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    augmentation = [
        (data / "augmented_3500" / (row["recording"] + ".gz"), row["recording"],
         [float(row[axis + "_m"]) for axis in "xyz"])
        for row in train
    ]
    augmented = append_training(original, augmentation, csi_features)
    coordinates = json.loads((REPO / "kaggle" / "scenario_coordinates.json").read_text(encoding="utf-8"))
    seven = append_training(
        augmented, [(data / "new_scenarios" / name, name, xyz)
                    for name, xyz in coordinates["seven"].items()], csi_features
    )
    protected = {row.tobytes() for row in original["X"][np.isin(original["split"], ["val", "test"])]}
    sixteen = append_training(
        seven, [(data / "newer discovered scenarious" / name, name, xyz)
                for name, xyz in coordinates["nine"].items()], csi_features, protected
    )
    results = {}
    test = load_data(original_path)
    for name, dataset, expected in [
        ("original", original, 17888), ("augmented", augmented, 59888),
        ("seven_scenarios", seven, 70388), ("sixteen_scenarios", sixteen, 91988),
    ]:
        n_train = int(np.sum(dataset["split"] == "train"))
        if n_train != expected:
            raise ValueError(f"{name}: expected {expected} training packets, got {n_train}")
        path = original_path if name == "original" else out / f"{name}.npz"
        if name != "original":
            np.savez_compressed(path, **dataset)
        run_dir = out / name
        subprocess.run(
            [sys.executable, "-u", str(code / "recover_train.py"), str(path),
             "--out", str(run_dir), "--init", str(checkpoint), "--epochs", "30",
             "--batch-size", "512", "--seed", "42", "--device", "cuda"], check=True
        )
        metrics, predictions, labels, recordings = score(
            read_checkpoint(run_dir / "best.pt"), test, torch.device("cuda"), use_amp=False
        )
        if len(predictions) != 5941 or len(set(recordings)) != 4:
            raise ValueError("Unexpected held-out test set.")
        np.savez_compressed(run_dir / "test_full_precision.npz", predictions=predictions,
                            labels=labels, recording=recordings)
        results[name] = {"training_packets": n_train, **metrics}
        (out / "comparison_full_precision.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(name, metrics["mean_distance_m"], flush=True)


if __name__ == "__main__":
    main()
