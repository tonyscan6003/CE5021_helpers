"""Plots of images, predictions, confusion matrices and TensorBoard logs.

All plots are drawn with matplotlib, so they are saved in the notebook and
visible when it is viewed on GitHub.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from .data import IMAGENET_MEAN, IMAGENET_STD, denormalize


def show_images(images, labels, class_names=None, predictions=None, n=12,
                mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Show the first ``n`` images of a batch in a row, titled with their class.

    If ``predictions`` are given, each title is the predicted class, in black
    when correct and red when wrong.
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 2.2))
    axes = [axes] if n == 1 else axes

    def name(index):
        index = int(index)
        return class_names[index] if class_names else str(index)

    for i, ax in enumerate(axes):
        ax.imshow(denormalize(images[i], mean, std))
        ax.axis("off")
        if predictions is None:
            ax.set_title(name(labels[i]), fontsize="small")
        else:
            correct = int(predictions[i]) == int(labels[i])
            ax.set_title(name(predictions[i]), fontsize="small",
                         color="black" if correct else "red")
    plt.show()


def show_batch(loader, class_names=None, n=12):
    """Show ``n`` images from the next batch of a data loader (after the transforms)."""
    images, labels = next(iter(loader))
    show_images(images, labels, class_names, n=n)


@torch.no_grad()
def show_predictions(model, loader, device, class_names=None, n=12):
    """Show ``n`` images with the model's predictions (wrong predictions in red)."""
    model.eval()
    images, labels = next(iter(loader))
    predictions = model(images.to(device)).argmax(dim=1).cpu()
    show_images(images, labels, class_names, predictions=predictions, n=n)


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
    """Plot a confusion matrix: rows are the true class, columns the predicted class.

    With ``normalize=True`` each row shows the fraction of that class
    predicted as each class (the diagonal is the per-class recall).
    """
    size = max(5, 0.6 * len(class_names)) if class_names else 6
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax,
        normalize="true" if normalize else None,
        values_format=".2f" if normalize else "d", colorbar=False)
    ax.set_title("Confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_tensorboard_logs(logdir="runs"):
    """Plot every scalar logged with a TensorBoard SummaryWriter under ``logdir``.

    Use this when training code logs to TensorBoard directly: TensorBoard
    itself is not visible in a notebook saved to GitHub, these plots are.
    Scalars are grouped by the part of the tag before "/" (e.g. "Loss/train"
    and "Loss/validation" share a plot), with one line per run folder.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    root = Path(logdir)
    event_dirs = sorted({p.parent for p in root.rglob("events.out.tfevents.*")})
    if not event_dirs:
        print(f"No TensorBoard logs found in '{logdir}'.")
        return

    groups = {}                      # {plot title: {line label: (steps, values)}}
    for event_dir in event_dirs:
        acc = EventAccumulator(str(event_dir))
        acc.Reload()
        run = event_dir.relative_to(root).as_posix()
        for tag in acc.Tags()["scalars"]:
            title, _, series = tag.partition("/")
            label = " ".join(part for part in (run if run != "." else "", series) if part)
            events = acc.Scalars(tag)
            groups.setdefault(title, {})[label or title] = (
                [e.step for e in events], [e.value for e in events])

    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4), squeeze=False)
    for ax, (title, lines) in zip(axes[0], groups.items()):
        for label, (steps, values) in lines.items():
            ax.plot(steps, values, label=label)
        ax.set(title=title, xlabel="step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    fig.tight_layout()
    plt.show()
