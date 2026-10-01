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
def evaluate_balanced(model, loader, loss_fn, device):
    """Like ``evaluate``, but also returns the balanced accuracy.

    Returns:
        (average loss, accuracy in %, balanced accuracy in %).

    Balanced accuracy is the average of the accuracy on each class (the
    per-class recall). When one class is much more common than the others,
    a model can get a high accuracy by mostly predicting that class; its
    balanced accuracy stays low unless the rare classes are recognised too.
    """
    from sklearn.metrics import balanced_accuracy_score

    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += loss_fn(outputs, labels).item() * len(labels)
        y_true.append(labels.cpu())
        y_pred.append(outputs.argmax(dim=1).cpu())
    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    return (total_loss / len(y_true), 100 * float((y_true == y_pred).mean()),
            100 * balanced_accuracy_score(y_true, y_pred))


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
        self.val_bal_acc = []

        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"{logdir}/{name}_{time.strftime('%Y%m%d-%H%M%S')}")

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr=None,
            val_bal_acc=None, verbose=True):
        """Record the results of one epoch (and print them unless verbose=False).

        ``val_bal_acc`` (optional) is the validation balanced accuracy; when it
        is given, it decides which epoch is the best.
        """
        self.epoch.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.lr.append(lr)
        if val_bal_acc is not None:
            self.val_bal_acc.append(val_bal_acc)

        if verbose:
            lr_text = f"  lr {lr:.2e}" if lr is not None else ""
            bal_text = f"  balanced {val_bal_acc:5.1f}%" if val_bal_acc is not None else ""
            best_text = "  *best" if self.is_best() else ""
            print(f"Epoch {epoch + 1:3d} | train loss {train_loss:.4f}  acc {train_acc:5.1f}% | "
                  f"val loss {val_loss:.4f}  acc {val_acc:5.1f}%{bal_text}{lr_text}{best_text}")

        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/validation", val_acc, epoch)
            if val_bal_acc is not None:
                self.writer.add_scalar("Accuracy/validation balanced", val_bal_acc, epoch)
            if lr is not None:
                self.writer.add_scalar("Learning rate", lr, epoch)
            self.writer.flush()

    def _score(self):
        return self.val_bal_acc if self.val_bal_acc else self.val_acc

    def best_index(self):
        """Index of the best epoch so far: highest validation balanced accuracy
        (or accuracy, if balanced accuracy is not logged). The first one wins a tie."""
        return int(np.argmax(self._score()))

    def is_best(self):
        """True if the epoch logged last is the best so far (use it to keep the best weights)."""
        return bool(self.epoch) and self.best_index() == len(self.epoch) - 1

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
        if self.val_bal_acc:
            axes[1].plot(epochs, self.val_bal_acc, "--", label="validation (balanced)")
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
        best = self.best_index()
        result = {
            "run": self.name,
            **{k: v for k, v in self.config.items() if k != "run_name"},
            "epochs trained": len(self.epoch),
            "final train acc": round(self.train_acc[-1], 1),
            "final val acc": round(self.val_acc[-1], 1),
            "best val acc": round(self.val_acc[best], 1),
            "best epoch": self.epoch[best] + 1,
        }
        if self.val_bal_acc:
            result["best val balanced acc"] = round(self.val_bal_acc[best], 1)
        width = max(len(k) for k in result)
        for key, value in result.items():
            print(f"{key:<{width}} : {value}")
        return result


@torch.no_grad()
def final_evaluation(model, history, train_loader, test_loader, device):
    """Evaluate the trained model once on the test set and print the result block.

    The training compute is recalculated here from the model itself (which
    layers were trained), the image size of the test images, the number of
    training images per epoch and the number of epochs in ``history``:

        training compute = training FLOPs per image x images per epoch x epochs

    Returns a dict with all values shown in the block.
    """
    import datetime

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    from .models import compute_cost, format_count

    model.eval()
    y_true, probs = [], []
    for inputs, labels in test_loader:
        probs.append(torch.softmax(model(inputs.to(device)), dim=1).cpu())
        y_true.append(torch.as_tensor(labels))
    y_true, probs = torch.cat(y_true).numpy(), torch.cat(probs).numpy()
    y_pred = probs.argmax(axis=1)
    if probs.shape[1] == 2:
        auc = roc_auc_score(y_true, probs[:, 1])
    else:
        auc = roc_auc_score(y_true, probs, multi_class="ovr")

    image_size = next(iter(test_loader))[0].shape[-1]
    cost = compute_cost(model, image_size, device, verbose=False)
    images_per_epoch = len(train_loader.sampler)
    epochs = len(history.epoch)
    best = history.best_index() if epochs else None
    compute = cost["train_flops"] * images_per_epoch * epochs
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)

    result = {
        "run": history.name,
        "model": cost.get("arch", type(model).__name__),
        "strategy": cost.get("strategy"),
        "head": cost.get("head"),
        "image size": image_size,
        "training images per epoch": images_per_epoch,
        "epochs trained": epochs,
        "weights from epoch": history.epoch[best] + 1 if epochs else None,
        "val balanced acc (%)": round(history._score()[best], 1) if epochs else None,
        "TEST balanced acc (%)": round(100 * balanced_accuracy_score(y_true, y_pred), 1),
        "test acc (%)": round(100 * float((y_true == y_pred).mean()), 1),
        "test AUC": round(float(auc), 3),
        "parameters": format_count(cost["params"]),
        "trained parameters": format_count(cost["trainable_params"]),
        "inference GFLOP per image": round(cost["inference_flops"] / 1e9, 3),
        "training GFLOP per image": round(cost["train_flops"] / 1e9, 3),
        "TRAINING COMPUTE (TFLOP)": round(compute / 1e12, 2),
        "settings": history.config,
        "evaluated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                     + f" on {gpu}",
    }
    width = max(len(k) for k in result)
    line = "=" * 72
    print(f"{line}\nFINAL RESULT (test set, evaluated once)\n{line}")
    for key, value in result.items():
        print(f"{key:<{width}} : {value}")
    print(line)
    return result
