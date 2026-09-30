"""Device selection, evaluation and recording of training results.

The training loop itself is written in the notebooks so it can be read and
changed (e.g. to add a learning-rate scheduler). These helpers cover the
parts that are the same in every assignment::

    device = get_device()
    history = History("baseline", config=config)

    for epoch in range(n_epochs):
        train_loss, train_acc = ...                        # your training code
        val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)
        history.log(epoch, train_loss, train_acc, val_loss, val_acc)

    history.plot()
    y_true, y_pred = predict(model, test_loader, device)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device():
    """Return "cuda" (NVIDIA GPU), "mps" (Apple GPU) or "cpu", whichever is available."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using {device} device")
    return device


@torch.no_grad()      # no gradients needed: we only measure, we don't train
def evaluate(model, loader, loss_fn, device):
    """Run the model over every batch in ``loader``.

    Returns:
        (average loss, accuracy in %) over all images in the loader.

    The model is left in evaluation mode (dropout off, batch-norm statistics
    frozen); call ``model.train()`` before training again.
    """
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += loss_fn(outputs, labels).item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, 100 * correct / seen


@torch.no_grad()
def predict(model, loader, device):
    """Return (y_true, y_pred) as numpy arrays of class indices for every image in ``loader``."""
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        y_pred.append(outputs.argmax(dim=1).cpu().numpy())
        y_true.append(np.asarray(labels))
    return np.concatenate(y_true), np.concatenate(y_pred)


class History:
    """Record loss, accuracy and learning rate for each epoch of one training run.

    Args:
        name: a short label for this run, e.g. "augment+onecycle".
        config: optional dict of the settings used (learning rate, epochs, ...);
            shown by ``summary()`` so each run can be added to a results table.
        tensorboard: also write the values to TensorBoard (folder
            ``runs/<name>_<time>``), for live monitoring while training.

    The recorded values are plain lists, e.g. ``history.val_acc[-1]`` is the
    validation accuracy after the last epoch.
    """

    def __init__(self, name="run", config=None, tensorboard=True, logdir="runs"):
        self.name = name
        self.config = dict(config or {})
        self.epoch, self.lr = [], []
        self.train_loss, self.train_acc = [], []
        self.val_loss, self.val_acc = [], []

        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"{logdir}/{name}_{time.strftime('%Y%m%d-%H%M%S')}")

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr=None, verbose=True):
        """Record the results of one epoch (and print them unless verbose=False)."""
        self.epoch.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.lr.append(lr)

        if verbose:
            lr_text = f"  lr {lr:.2e}" if lr is not None else ""
            print(f"Epoch {epoch + 1:3d} | train loss {train_loss:.4f}  acc {train_acc:5.1f}% | "
                  f"val loss {val_loss:.4f}  acc {val_acc:5.1f}%{lr_text}")

        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/validation", val_acc, epoch)
            if lr is not None:
                self.writer.add_scalar("Learning rate", lr, epoch)
            self.writer.flush()

    def plot(self):
        """Plot loss and accuracy curves (and the learning rate, if it was logged).

        Unlike TensorBoard, these plots are saved in the notebook, so they are
        visible when the notebook is viewed on GitHub.
        """
        if not self.epoch:
            print("Nothing to plot: no epochs have been logged yet.")
            return
        epochs = [e + 1 for e in self.epoch]
        has_lr = any(lr is not None for lr in self.lr)
        fig, axes = plt.subplots(1, 3 if has_lr else 2, figsize=(15 if has_lr else 11, 4))

        axes[0].plot(epochs, self.train_loss, label="train")
        axes[0].plot(epochs, self.val_loss, label="validation")
        axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")

        axes[1].plot(epochs, self.train_acc, label="train")
        axes[1].plot(epochs, self.val_acc, label="validation")
        axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy (%)")

        if has_lr:
            axes[2].plot(epochs, self.lr, color="tab:green")
            axes[2].set(title="Learning rate", xlabel="epoch", ylabel="learning rate",
                        yscale="log")

        for ax in axes:
            ax.grid(alpha=0.3)
        for ax in axes[:2]:
            ax.legend()
        fig.suptitle(self.name)
        fig.tight_layout()
        plt.show()

    def summary(self):
        """Print (and return as a dict) the key results of this run."""
        if not self.epoch:
            print("No epochs logged yet.")
            return {}
        best = int(np.argmax(self.val_acc))
        result = {
            "run": self.name,
            **self.config,
            "epochs trained": len(self.epoch),
            "final train acc": round(self.train_acc[-1], 1),
            "final val acc": round(self.val_acc[-1], 1),
            "best val acc": round(self.val_acc[best], 1),
            "best epoch": self.epoch[best] + 1,
        }
        width = max(len(k) for k in result)
        for key, value in result.items():
            print(f"{key:<{width}} : {value}")
        return result
