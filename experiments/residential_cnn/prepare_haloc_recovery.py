"""Build a labeled HALOC feature archive from released CSVs or the team's compact Parquet."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from prepare_captures import csi_features

URL = "https://zenodo.org/records/10715595/files/HALOC.zip?download=1"
EXPECTED_MD5 = "f3bf1e60cc2f0d1bd6e6a75cce247437"
PARTITIONS = ("train", "train", "train", "train", "val", "test")


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare(source, output, order):
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".parquet":
        import pandas as pd
        data = pd.read_parquet(source)
        columns = [f"{kind}_{i}" for kind in ("amp", "phase") for i in range(128)]
        required = columns + ["x", "y", "z", "split"]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Compact file is missing {sorted(missing)}")
        x = data[columns].to_numpy(dtype=np.float32)
        y = data[["x", "y", "z"]].to_numpy(dtype=np.float32)
        split = np.asarray(data["split"].astype(str).tolist(), dtype=str)
        recordings = np.char.add("HALOC-", split.astype(str))
        provenance = {
            "source": source.name, "sha256": digest(source),
            "feature_convention": "precomputed team features",
            "declared_pair_order": order,
        }
    else:
        groups, labels, names, partitions = [], [], [], []
        checksums = {}
        for index, partition in enumerate(PARTITIONS):
            filename = source / f"{index}.csv"
            checksums[filename.name] = digest(filename)
            count = 0
            with filename.open(newline="", encoding="utf-8-sig") as stream:
                for line, row in enumerate(csv.DictReader(stream), 2):
                    raw = np.asarray(json.loads(row["data"]), dtype=np.float32)
                    target = np.asarray([row[axis] for axis in "xyz"], dtype=np.float32)
                    if raw.shape != (256,) or not np.isfinite(raw).all() or not np.isfinite(target).all():
                        raise ValueError(f"Invalid HALOC values in {filename.name}, line {line}")
                    groups.append(csi_features(raw, order))
                    labels.append(target)
                    names.append(filename.name)
                    partitions.append(partition)
                    count += 1
            if not count:
                raise ValueError(f"Empty HALOC sequence {filename}")
            print(f"{filename.name}: {count:,} packets, {partition}", flush=True)
        x, y = np.asarray(groups, dtype=np.float32), np.asarray(labels, dtype=np.float32)
        split, recordings = np.asarray(partitions), np.asarray(names)
        provenance = {
            "source": URL, "sequence_sha256": checksums,
            "feature_convention": "fresh CSV conversion", "declared_pair_order": order,
        }
    if set(split) != {"train", "val", "test"} or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("HALOC must contain finite features and all three partitions")
    np.savez_compressed(output, X=x, Y=y, split=split, recording=recordings, byte_order=order)
    provenance["partition_packets"] = {name: int(np.sum(split == name)) for name in PARTITIONS}
    provenance["total_packets"] = len(x)
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def download(destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "HALOC.zip"
    if not archive.exists() or digest(archive, "md5") != EXPECTED_MD5:
        temporary = destination / "HALOC.zip.partial"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(URL, timeout=90) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if digest(temporary, "md5") != EXPECTED_MD5:
                    raise ValueError("Downloaded HALOC archive failed its checksum")
                temporary.replace(archive)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
    output = destination / "HALOC"
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for index in range(6):
            name = f"{index}.csv"
            (output / name).write_bytes(zipped.read(f"HALOC/{name}"))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Team compact Parquet or directory of six official CSVs")
    parser.add_argument("--download-dir", type=Path, default=Path("work/external"))
    parser.add_argument("--out", type=Path, default=Path("work/haloc.npz"))
    parser.add_argument("--byte-order", choices=("iq", "qi"), default="iq")
    args = parser.parse_args()
    source = args.source if args.source else download(args.download_dir)
    prepare(source, args.out, args.byte_order)
