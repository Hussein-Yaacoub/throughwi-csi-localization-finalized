# ThroughWi Localization

ThroughWi is an American University of Beirut electrical and computer engineering final-year project. This repository accompanies *ThroughWi: Single-Link Wi-Fi Sensing for Stationary-Person Localization in Homes*. It includes the ESP32 capture workflow, residential recordings, and the paired CNN training and evaluation experiment.

**Team** · Hussein Yaacoub, Hassan Khalil, Jihad Mobarak, and Ali Assaad

**Advisor** · Dr. Fadi Zaraket

## Project achievements

- An ESP32 and commodity Wi-Fi access point provide device-free sensing, with active packet triggering and host-side serial logging.
- The acquisition check recorded 1,500 CSI frames in 150 seconds, or 10 frames/s. The earlier passive implementation obtained approximately 0.2 frames/s.
- The residential campaign comprises 29 recordings across two homes, with 43,147 cleaned packets and 26,812 position-labeled packets for localization.
- Residential fine-tuning reduced mean packet-level horizontal localization error by **44.4%** in the paired Multi-Scale CNN experiment.

## Latest Kaggle comparison

All four Multi-Scale CNN runs started from the same HALOC checkpoint and used the same four held-out residential recordings (5,941 packets). The added files were used for training. Checkpoints were selected on the original validation recordings, with seed 42 and 30 epochs for each condition.

| Residential training data | Training packets | Mean horizontal test error |
| --- | ---: | ---: |
| Original training captures | 17,888 | 1.161 m |
| + Augmentation from the training captures | 59,888 | 1.033 m |
| + First seven supplemental scenarios | 70,388 | 0.820 m |
| + Nine further supplemental scenarios | 91,988 | **0.766 m** |

The final condition reduced mean test error by 34.0% relative to the original-training-data condition, and by 6.6% relative to the seven-scenario condition. These are single-seed results on the original residential test split. Scenario coordinates were inferred from filenames. Git history describes the nine newer scenarios as interpolated.

See [Kaggle artifacts and reproduction instructions](kaggle/README.md) and the [data inventory](data/README.md).

## Earlier paired fine-tuning experiment

| Predictor | Mean horizontal error |
| --- | ---: |
| Pre-fine-tuning CNN baseline (HALOC-trained, with residential training normalization) | 2.145 m |
| CNN after residential fine-tuning | **1.193 m** |

Both CNN checkpoints were evaluated on the same four held-out recordings (5,941 packets) from setups represented in training. Both CNN checkpoints used input and coordinate normalization fitted on the residential training split. Fine-tuning reduced error relative to the frozen CNN on all four recordings.

See the [saved results and per-recording breakdown](results/residential_cnn/README.md) and the [commands to reproduce the paired experiment](experiments/residential_cnn/README.md).

## Repository contents

| Directory | Contents |
| --- | --- |
| [firmware](firmware/) and [capture](capture/) | ESP32 acquisition firmware and Python serial logger |
| [data](data/) | 29 original residential recordings, their partition manifest, 29 augmentation-only files with 3,500 rows each, and sixteen supplemental scenarios |
| [notebooks/historical](notebooks/historical/) | Original experiment notebooks |
| [scripts](scripts/) | Preparation, training, fine-tuning, and supplementary evaluation utilities |
| [experiments/residential_cnn](experiments/residential_cnn/) | Paired CNN training and evaluation workflow |
| [results](results/) | Archived CNN results |
| [kaggle](kaggle/) | Latest CNN comparison, checkpoints, and reproduction instructions |

The [experiment notes](docs/experiments.md) describe the manuscript workflow, standalone utilities, and historical notebooks.

## Prepare the residential data

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_captures.py --byte-order iq
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Preparation reads `data/recordings.csv` and the compressed capture files. It writes `work/captures.npz` (features, coordinates, recording identifiers, partition labels, and byte order) and `work/captures_audit.json` (per-recording cleaning counts). The explicit `iq` option matches the pairing convention in the original residential cleaning notebook.

## Standalone training utilities

Download the six HALOC CSVs from the [official dataset record](https://zenodo.org/records/10715595) into `data/external/HALOC/`. HALOC is distributed separately under its own terms.

The standalone runner accepts the prepared NPZ file and supports `--model multiscale`, `--model transformer`, and `--model se_resnet`. Optional `--init` loads a checkpoint produced by this runner with matching architecture and byte order. Normalization is fitted on training packets, and validation mean Euclidean error selects the checkpoint.

| Runner setting | Current script behavior |
| --- | --- |
| Epochs | 30 by default; configurable with `--epochs` |
| Batch size | 128 by default; configurable with `--batch-size` |
| Optimizer and schedule | AdamW and cosine annealing |
| Learning rate | 0.0005 with initialization or for Transformer; otherwise 0.001; configurable with `--lr` |
| Test evaluation | Runs only when `--test` is supplied |
| Output directory | Required through `--out`; must be a new directory |
| Saved files | `best.pt` contains model weights and normalization statistics; `metrics.json` contains run settings and training history, plus test metrics when requested |

To reproduce the manuscript results, use the [paired CNN workflow](experiments/residential_cnn/README.md), which uses recorded training settings and separate evaluation commands.

See [capture setup](docs/capture.md) for flashing the board and collecting data, and [data documentation](data/README.md) for the recording inventory.
