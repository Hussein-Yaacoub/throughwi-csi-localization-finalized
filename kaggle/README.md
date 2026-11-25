# Kaggle residential CNN comparison

| Condition | Training packets | Full precision mean horizontal test error |
| --- | ---: | ---: |
| Original | 17,888 | 1.160891652 m |
| Augmented | 59,888 | 1.032812595 m |
| Seven scenarios | 70,388 | 0.819761693 m |
| Sixteen scenarios | 91,988 | 0.765831947 m |

Every condition used the same 2,983 validation packets and 5,941 test packets from four held-out recordings. All started from one HALOC checkpoint, used 30 epochs, batch size 512 and seed 42, and selected the checkpoint using validation error. These runs measure performance within the original residential test split. Supplemental coordinates are nominal labels inferred from filenames. Git history describes the nine newer scenarios as interpolated. These files contained 2,400 usable rows each and no exact preprocessed packet matches against the validation or test split.

The [comparison](comparison.json) records the first two conditions. The [latest report](sixteen_scenarios/comparison_full_precision.json) records scenario counts, coordinates, commits and full precision results for the final two conditions. Checkpoint folders contain the selected weights, normalization, training settings and history, split identifiers, and aligned predictions.

## Reproduce

Use a CUDA environment with Python, NumPy and PyTorch. The Kaggle runs used a Tesla T4 with PyTorch 2.11.0+cu130 and NumPy 2.1.3. From the repository root run:

```sh
python kaggle/run_comparison.py --out work/kaggle-comparison
```

Training code is pinned to commit `af54c0cdd1e2be3a0d0c0f524ebef15db50bf87e` and the data to `6bff28e8aec35c85365b95ffc1d5d0631245b973`. The bundled source checkpoint matches SHA-256 `a77dda823d02c2d900a5908972b1752be1d1dc4866a48cb032ded29273ea52c7`. See the [earlier paired experiment](../experiments/residential_cnn/README.md) for HALOC source training.
