# Residential CNN experiment

This workflow reproduces the paired Multi-Scale CNN evaluation reported in the ThroughWi manuscript. It uses the public HALOC CSVs and the repository's residential recording split. Both checkpoints are evaluated on the same four held-out residential recordings (5,941 packets) with input and coordinate normalization fitted on the residential training split.

Mean horizontal localization error fell from **2.145 m** before fine-tuning to **1.193 m** after fine-tuning, a **44.4%** reduction. See the [saved results](../../results/residential_cnn/README.md) for both checkpoints' metrics and the per-recording comparison before and after fine-tuning.

## Run from the repository root

The recorded run used Python 3.10, NumPy 2.2.6, and PyTorch 2.3.1 with CUDA 12.1. Use an existing CUDA environment or install the CUDA 12.1 wheel compatible with your system.

```sh
python -m pip install numpy==2.2.6
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Run these commands from the repository root. HALOC preparation downloads the official archive and verifies its checksum. To use an existing extraction, provide `--source` with the path to the six official CSVs.

```sh
python experiments/residential_cnn/prepare_haloc_recovery.py --download-dir work/residential_cnn/external --out work/residential_cnn/haloc.npz --byte-order iq
python experiments/residential_cnn/prepare_captures.py --data data --out work/residential_cnn --byte-order iq
python experiments/residential_cnn/recover_train.py work/residential_cnn/haloc.npz --out work/residential_cnn/source-120 --epochs 120 --batch-size 2048 --seed 42 --device cuda
python experiments/residential_cnn/recover_train.py work/residential_cnn/captures.npz --out work/residential_cnn/residential-30 --init work/residential_cnn/source-120/best.pt --epochs 30 --batch-size 512 --seed 42 --device cuda
python experiments/residential_cnn/cross_evaluate.py --source-data work/residential_cnn/haloc.npz --residential-data work/residential_cnn/captures.npz --pretrained work/residential_cnn/source-120/best.pt --finetuned work/residential_cnn/residential-30/best.pt --out work/residential_cnn/comparison
```

Source training uses OneCycle scheduling and residential fine-tuning uses cosine annealing. Both runs select the best checkpoint on their validation partition. The evaluator uses full-precision inference and writes aggregate metrics and aligned packet predictions. For the residential comparison, both checkpoints use residential training normalization. See the [original project protocol and standalone utilities](../../docs/experiments.md) for the other workflows.

Data source: [HALOC on Zenodo](https://zenodo.org/records/10715595).
