# Experiment documentation

## Paired residential CNN experiment

The manuscript reports one Multi-Scale CNN trained on public HALOC data and evaluated on the same residential test set before and after fine-tuning. Both checkpoints use residential training-set normalization at evaluation. Mean horizontal error fell from **2.145 m** to **1.193 m**, a **44.4%** reduction across four held-out recordings (5,941 packets).

The [reproduction workflow](../experiments/residential_cnn/README.md) records the commands for preprocessing, training, and evaluation. The [run records](../results/residential_cnn/README.md) give settings, checkpoint epochs, and per-recording results.

## Historical notebooks

The [original project notebooks](../notebooks/historical/) preserve the final-year project implementations. The manuscript's paired CNN results come from the experiment above.

## Standalone utilities

The standalone preparation and training scripts were added during repository organization. They are distinct from the original notebook execution.

- Residential preparation defaults to Q/I byte order. Use `--byte-order iq` to match the original cleaner's I/Q interpretation. Swapping the order preserves amplitudes and changes phases.
- The HALOC converter reads the official per-packet JSON and XYZ fields. It uses sequence assignments 0–3 for training, 4 for validation, and 5 for testing, covering 96,491 / 28,111 / 14,277 packets. It is a separate implementation from the original compact feature export.
- The training runner uses cosine scheduling for both stages and defaults to batch size 128. The original source notebook uses OneCycleLR and batch size 2,048.
- The runner defaults to a fine-tuning learning rate of 0.0005 for all architectures. Pass `--lr 0.0003` for the original Transformer fine-tuning learning rate.
- The runner's scaling replaces near-zero standard deviations with 1.0; the notebooks add a small epsilon to standard deviations.
- The runner saves weights with normalization statistics and configuration, and selects weights by validation mean Euclidean distance. Test evaluation is enabled with `--test`.

For the available command-line options:

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_haloc.py --help
python scripts/train.py --help
```

Use consistent feature conventions and matching architectures when initializing from another run. The standalone defaults are not an exact replay of the original training configuration.
