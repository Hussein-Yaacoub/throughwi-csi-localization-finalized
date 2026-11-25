# Residential CNN experiment results

We evaluated the HALOC-trained checkpoint and the residentially fine-tuned checkpoint on the same four held-out residential recordings, totaling 5,941 packets. Both used input and coordinate normalization fitted on the residential training recordings.

| Predictor | Mean horizontal error |
| --- | ---: |
| Pre-fine-tuning CNN baseline (HALOC-trained, with residential training normalization) | 2.145 m |
| CNN after residential fine-tuning | **1.193 m** |

Fine-tuning reduced mean horizontal error by **44.4%**, calculated from the full-precision values. All test setups were represented in training.

| Test position (x, y), m | Packets | Before fine-tuning (m) | After fine-tuning (m) |
| --- | ---: | ---: | ---: |
| (2, 0), same room | 1,470 | 1.871 | 0.764 |
| (1, 1), same room | 1,482 | 1.553 | 0.984 |
| (2, −1), same room | 1,494 | 2.268 | 1.391 |
| (3.5, 0), two walls | 1,495 | 2.879 | 1.626 |

The pretrained checkpoint also scored **2.676 m** mean Euclidean error on the separate HALOC test sequence using HALOC training normalization.

- [`cross_evaluation.json`](cross_evaluation.json) includes per-recording errors, checkpoint settings, and cross-domain diagnostics. Read `pretrained_with_residential_calibration` and `finetuned_on_residential` for the residential comparison. The entries using the checkpoint's native normalization on the other domain are diagnostic because nearly constant features can produce extreme values after standardization.
- [`source-120/`](source-120/) and [`residential-30/`](residential-30/) contain the training settings, epoch histories, and split information.
- [`checkpoint_sha256.json`](checkpoint_sha256.json) identifies the paired experiment checkpoints by SHA-256.
- [Reproduction commands](../../experiments/residential_cnn/README.md) generate checkpoints and packet predictions.
