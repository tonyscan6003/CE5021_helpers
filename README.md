# ce5021 — helper functions for the CE5021 assignments

Code shared by the CE5021 notebooks: data loading, evaluation, recording
training results and plotting. The training loops themselves stay in the
notebooks so you can read and change them.

## Install

In Google Colab (the first cell of each assignment does this for you):

```python
%pip install -q "git+https://github.com/tonyscan6003/CE5021_helpers@ay2627"
```

## Contents

| Module | Functions |
|---|---|
| `ce5021.data` | `cifar10_loaders`, `medmnist_loaders`, `denormalize`, `IMAGENET_MEAN`, `IMAGENET_STD`, `CIFAR10_CLASSES` |
| `ce5021.train` | `get_device`, `evaluate`, `predict`, `History` (`log`, `plot`, `summary`) |
| `ce5021.plots` | `show_batch`, `show_images`, `show_predictions`, `plot_confusion_matrix`, `plot_tensorboard_logs` |

In a notebook, `help(evaluate)` shows the documentation of a function and
`evaluate??` shows its source code.
