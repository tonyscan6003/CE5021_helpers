"""Semantic segmentation: the ATR human parsing dataset, pre-trained encoder stages,
evaluation (IoU) and plots.

The network and the training loop are written in the notebook::

    train_loader, val_loader, test_loader = atr_loaders(image_size, train_tf, test_tf)
    logits = model(images)                    # [B, n_classes, H, W], same H, W as the images
    loss = loss_fn(logits, masks)             # masks: [B, H, W] class index of every pixel

    metrics = evaluate_segmentation(model, val_loader, loss_fn, device)   # loss, mIoU, pixel acc
    preds, masks = predict_segmentation(model, val_loader, device)
    show_segmentation(val_loader.dataset, preds, indices=worst_images(preds, masks))

``image_size`` is ``(height, width)``; both must be multiples of 32.
"""

import datetime
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import IMAGENET_MEAN, IMAGENET_STD, _download, denormalize

# The ATR dataset (Liang et al., "Deep Human Parsing with Active Template Regression", 2015):
# 17,706 photographs of one person each, every pixel labelled with one of 18 classes.
ATR_URL = ("https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset/resolve/main/"
           "data/train-0000{i}-of-00002-{suffix}.parquet")
ATR_FILES = ("f3a663f7140ee7fd", "74610e243c32d5d6")
ATR_SIZE = 17_706
ATR_SPLIT = (14_706, 1_000, 2_000)      # training, validation, test images
ATR_CLASSES = ["background", "hat", "hair", "sunglasses", "upper-clothes", "skirt", "pants",
               "dress", "belt", "left-shoe", "right-shoe", "face", "left-leg", "right-leg",
               "left-arm", "right-arm", "bag", "scarf"]

# Class of each class after a horizontal flip: a left shoe becomes a right shoe, etc.
_FLIP_CLASSES = list(range(len(ATR_CLASSES)))
for _left, _right in ((9, 10), (12, 13), (14, 15)):
    _FLIP_CLASSES[_left], _FLIP_CLASSES[_right] = _right, _left

# One colour per class for the plots (background black).
CLASS_COLOURS = np.array([(0, 0, 0)] + [tuple(int(255 * c) for c in plt.cm.tab20(i)[:3])
                                        for i in range(len(ATR_CLASSES) - 1)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def download_atr(root="data"):
    """Download the ATR parquet files from Hugging Face (first time only, about 800 MB).

    Returns the list of downloaded files.
    """
    folder = os.path.join(root, "atr")
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, suffix in enumerate(ATR_FILES):
        path = os.path.join(folder, f"train-{i}.parquet")
        if not os.path.exists(path):
            print(f"Downloading ATR part {i + 1} of {len(ATR_FILES)} (about 400 MB) ...")
            _download(ATR_URL.format(i=i, suffix=suffix), path)
        paths.append(path)
    return paths


def prepare_atr(image_size, root="data"):
    """Resize every ATR image and mask to ``image_size`` once and store them as arrays.

    Returns the paths of ``images`` [N, H, W, 3] and ``masks`` [N, H, W] (uint8
    ``.npy`` files, read from disk while training rather than held in memory).
    Images are resized smoothly; masks with the nearest pixel, so every pixel
    keeps a valid class.
    """
    import pyarrow.parquet as pq     # preinstalled in Colab
    from PIL import Image

    height, width = image_size
    folder = os.path.join(root, "atr")
    images_path = os.path.join(folder, f"images_{height}x{width}.npy")
    masks_path = os.path.join(folder, f"masks_{height}x{width}.npy")
    if os.path.exists(images_path) and os.path.exists(masks_path):
        return images_path, masks_path

    parquet_files = download_atr(root)
    print(f"Resizing the images to {height} x {width} (first time only, about a minute) ...")
    images = np.lib.format.open_memmap(images_path + ".part", mode="w+", dtype=np.uint8,
                                       shape=(ATR_SIZE, height, width, 3))
    masks = np.lib.format.open_memmap(masks_path + ".part", mode="w+", dtype=np.uint8,
                                      shape=(ATR_SIZE, height, width))
    n = 0
    for path in parquet_files:
        table = pq.read_table(path)
        for start in range(0, table.num_rows, 1000):
            for row in table.slice(start, 1000).to_pylist():
                image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                mask = Image.open(io.BytesIO(row["mask"]["bytes"]))
                images[n] = np.asarray(image.resize((width, height), Image.BILINEAR))
                masks[n] = np.asarray(mask.resize((width, height), Image.NEAREST))
                n += 1
    if n != ATR_SIZE or masks.max() >= len(ATR_CLASSES):
        raise ValueError(f"unexpected contents in {folder}: {n} images")
    images.flush(), masks.flush()
    del images, masks
    os.replace(images_path + ".part", images_path)
    os.replace(masks_path + ".part", masks_path)
    return images_path, masks_path


class RandomFlipLeftRight(nn.Module):
    """Flip an image and its mask horizontally with probability ``p``, and swap left and right.

    After a flip, a left arm is on the right of the image and looks like a
    right arm, so its label must change too: ``v2.RandomHorizontalFlip`` would
    keep the old labels and teach the network contradicting targets.
    Used inside ``v2.Compose`` with an (image, mask) pair.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.register_buffer("flip_classes", torch.tensor(_FLIP_CLASSES), persistent=False)

    def forward(self, image, mask):
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as TF
        if torch.rand(1).item() < self.p:
            image = TF.horizontal_flip(image)
            mask = tv_tensors.wrap(self.flip_classes[TF.horizontal_flip(mask).long()], like=mask)
        return image, mask


class ATRSegmentation(Dataset):
    """ATR images and masks (already resized), selected by ``indices``.

    Each item is ``(image, mask)``: the image after ``transform`` (a
    torchvision ``v2`` transform applied to the image and the mask together)
    and the mask as a tensor [H, W] of class indices (int64). Geometric
    transforms move the mask with the image; colour transforms only change
    the image.
    """

    def __init__(self, images_path, masks_path, indices, transform=None):
        self.images_path, self.masks_path = images_path, masks_path
        self.indices, self.transform = np.asarray(indices), transform
        self._images = self._masks = None
        self.classes = ATR_CLASSES

    def __len__(self):
        return len(self.indices)

    def class_pixel_counts(self):
        """Number of pixels of each class over all (untransformed) masks of this dataset."""
        masks = np.load(self.masks_path, mmap_mode="r")
        counts = np.zeros(len(ATR_CLASSES), dtype=np.int64)
        for start in range(0, len(self.indices), 1000):           # in chunks, to keep memory low
            chunk = masks[np.sort(self.indices[start:start + 1000])]
            counts += np.bincount(chunk.reshape(-1), minlength=len(ATR_CLASSES))
        return counts

    def __getitem__(self, index):
        from torchvision import tv_tensors
        if self._images is None:          # opened here, so each data-loading worker opens its own
            self._images = np.load(self.images_path, mmap_mode="r")
            self._masks = np.load(self.masks_path, mmap_mode="r")
        i = self.indices[index]
        image = tv_tensors.Image(torch.from_numpy(np.array(self._images[i])).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(np.array(self._masks[i])))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, torch.as_tensor(mask).long()


def atr_loaders(image_size, train_transform, test_transform, batch_size=16, n_train=None,
                root="data", split=ATR_SPLIT, seed=0, num_workers=2):
    """Download ATR and return (train_loader, val_loader, test_loader).

    Args:
        image_size: (height, width) of the images, both multiples of 32. The
            photographs are about 600 x 400 pixels (portrait); (288, 192) and
            (384, 256) keep their shape.
        n_train: number of training images to use (None: all ``split[0]``),
            for shorter experiments. The validation and test sets are always
            complete.

    The 17,706 images are split at random (fixed by ``seed``) into ``split``
    training, validation and test images. A batch is ``(images, masks)``:
    images [B, 3, H, W] and masks [B, H, W] holding the class index of every
    pixel.
    """
    height, width = image_size
    if height % 32 or width % 32:
        raise ValueError(f"image_size must be (height, width), both multiples of 32, not {image_size}")
    images_path, masks_path = prepare_atr(image_size, root)
    order = np.random.default_rng(seed).permutation(ATR_SIZE)
    n_tr, n_val, n_test = split
    parts = [order[:n_tr][:n_train], order[n_tr:n_tr + n_val], order[n_tr + n_val:n_tr + n_val + n_test]]
    datasets = [ATRSegmentation(images_path, masks_path, idx, tf)
                for idx, tf in zip(parts, (train_transform, test_transform, test_transform))]
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(datasets[0], shuffle=True, drop_last=True, **kwargs),
            DataLoader(datasets[1], **kwargs), DataLoader(datasets[2], **kwargs))


# ---------------------------------------------------------------------------
# Pre-trained encoder
# ---------------------------------------------------------------------------

ENCODERS = ("resnet18", "resnet34", "resnet50")


def resnet_stages(arch="resnet18", pretrained=True):
    """The stages of an ImageNet-trained ResNet, as separate modules, for an encoder.

    Returns ``(stages, channels)``: ``stages`` is an ``nn.ModuleList`` of 5
    modules applied one after the other; the output of stage k has
    ``channels[k]`` channels and 1/2, 1/4, 1/8, 1/16, 1/32 of the input size::

        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)     # kept for the skip connections of a decoder

    The classifier (average pooling and fully connected layer) is not included.
    """
    from torchvision import models as tv_models
    if arch not in ENCODERS:
        raise ValueError(f"arch must be one of {ENCODERS}, not {arch!r}")
    net = getattr(tv_models, arch)(weights="DEFAULT" if pretrained else None)
    stages = nn.ModuleList([
        nn.Sequential(net.conv1, net.bn1, net.relu),     # 1/2
        nn.Sequential(net.maxpool, net.layer1),          # 1/4
        net.layer2,                                      # 1/8
        net.layer3,                                      # 1/16
        net.layer4,                                      # 1/32
    ])
    expansion = 4 if arch == "resnet50" else 1
    channels = [64, 64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]
    return stages, channels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def confusion_matrix(preds, masks, n_classes=len(ATR_CLASSES)):
    """Pixel counts [n_classes, n_classes]: row = true class, column = predicted class."""
    preds = torch.as_tensor(np.asarray(preds)).reshape(-1).long()
    masks = torch.as_tensor(np.asarray(masks)).reshape(-1).long()
    return torch.bincount(masks * n_classes + preds, minlength=n_classes ** 2).reshape(
        n_classes, n_classes).numpy()


def metrics_from_confusion(conf):
    """Pixel accuracy, mean IoU and the IoU of each class (all in %) from a confusion matrix.

    The IoU of a class is (pixels correctly given the class) / (pixels that
    are the class or are predicted as the class). The mean IoU (mIoU) is the
    average over all classes, so a rare class counts as much as the background.
    """
    conf = conf.astype(np.float64)
    hits = np.diag(conf)
    union = conf.sum(axis=0) + conf.sum(axis=1) - hits
    iou = np.where(union > 0, hits / np.maximum(union, 1), np.nan)
    return {"pixel_acc": 100 * hits.sum() / max(conf.sum(), 1),
            "mIoU": 100 * float(np.nanmean(iou)),
            "class_iou": {name: 100 * float(v) for name, v in zip(ATR_CLASSES, iou)}}


def _check_output(logits, masks):
    if logits.shape[-2:] != masks.shape[-2:]:
        raise ValueError(f"the network output {tuple(logits.shape)} must have the same height and "
                         f"width as the images {tuple(masks.shape[-2:])}")


@torch.no_grad()
def evaluate_segmentation(model, loader, loss_fn, device):
    """Average loss, pixel accuracy, mIoU and class IoU (in %) of the model on ``loader``."""
    model.eval()
    total_loss, seen = 0.0, 0
    conf = np.zeros((len(ATR_CLASSES), len(ATR_CLASSES)), dtype=np.int64)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        _check_output(logits, masks)
        total_loss += loss_fn(logits, masks).item() * len(masks)
        seen += len(masks)
        conf += confusion_matrix(logits.argmax(dim=1).cpu(), masks.cpu())
    return {"loss": total_loss / seen, **metrics_from_confusion(conf)}


@torch.no_grad()
def predict_segmentation(model, loader, device):
    """Return (preds, masks): predicted and true class of every pixel, uint8 arrays [N, H, W]."""
    model.eval()
    preds, masks = [], []
    for images, true in loader:
        logits = model(images.to(device))
        _check_output(logits, true)
        preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
        masks.append(true.numpy().astype(np.uint8))
    return np.concatenate(preds), np.concatenate(masks)


def image_miou(pred, mask):
    """Mean IoU (%) of one image, over the classes that appear in its mask or prediction."""
    ious = []
    for c in np.union1d(np.unique(pred), np.unique(mask)):
        p, m = pred == c, mask == c
        ious.append((p & m).sum() / (p | m).sum())
    return 100 * float(np.mean(ious))


def worst_images(preds, masks, n=6):
    """Indices of the ``n`` images with the lowest mean IoU."""
    scores = np.array([image_miou(p, m) for p, m in zip(preds, masks)])
    return [int(i) for i in np.argsort(scores, kind="stable")[:n]]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def colourise(mask):
    """Class indices [H, W] -> an RGB image [H, W, 3] with one colour per class."""
    return CLASS_COLOURS[np.asarray(mask, dtype=np.int64)]


def _legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[i] / 255) for i in range(len(ATR_CLASSES))]
    fig.legend(handles, ATR_CLASSES, loc="lower center", ncol=9, fontsize="small", frameon=False)


def show_segmentation(dataset, preds=None, indices=None, n=4, title=""):
    """Show images with their true masks (and predicted masks, if ``preds`` is given).

    ``preds`` comes from ``predict_segmentation`` with a loader over ``dataset``
    that does not shuffle (validation or test), so it is in the same order as
    the dataset. The mean IoU of each image is shown above its prediction.
    """
    indices = list(range(min(n, len(dataset)))) if indices is None else list(indices)[:n]
    rows = 3 if preds is not None else 2
    fig, axes = plt.subplots(rows, len(indices), figsize=(2.6 * len(indices), 3.6 * rows + 0.6),
                             squeeze=False)
    for col, i in enumerate(indices):
        image, mask = dataset[i]
        axes[0, col].imshow(denormalize(image))
        axes[0, col].set_title(f"image {i}", fontsize="small")
        axes[1, col].imshow(colourise(mask), interpolation="nearest")
        axes[1, col].set_title("true classes", fontsize="small")
        if preds is not None:
            axes[2, col].imshow(colourise(preds[i]), interpolation="nearest")
            axes[2, col].set_title(f"predicted (mIoU {image_miou(preds[i], mask.numpy()):.0f}%)",
                                   fontsize="small")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()


def plot_class_pixels(dataset):
    """Bar chart of the share of pixels of each class in ``dataset`` (log scale)."""
    counts = dataset.class_pixel_counts()
    share = 100 * counts / counts.sum()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    bars = ax.bar(ATR_CLASSES, share, color=CLASS_COLOURS[np.arange(len(ATR_CLASSES))] / 255,
                  edgecolor="gray")
    ax.bar_label(bars, labels=[f"{s:.1f}" if s >= 0.1 else f"{s:.2f}" for s in share], fontsize="x-small")
    ax.set(yscale="log", ylabel="share of pixels (%)",
           title=f"Pixels of each class ({len(dataset)} images)")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    plt.show()
    return dict(zip(ATR_CLASSES, share))


def plot_class_iou(metrics, title="IoU of each class"):
    """Bar chart of the IoU of each class, with the mean IoU as a line."""
    names, values = list(metrics["class_iou"]), list(metrics["class_iou"].values())
    fig, ax = plt.subplots(figsize=(11, 3.4))
    bars = ax.bar(names, values, color=CLASS_COLOURS[np.arange(len(names))] / 255, edgecolor="gray")
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], fontsize="x-small")
    ax.axhline(metrics["mIoU"], color="tab:red", linestyle="--", label=f"mIoU {metrics['mIoU']:.1f}%")
    ax.set(ylim=(0, 105), ylabel="IoU (%)", title=title)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_confusion(preds, masks):
    """Confusion matrix of the pixels, each row divided by its total (recall of each class).

    Large values off the diagonal show which classes are mistaken for which.
    """
    conf = confusion_matrix(preds, masks).astype(np.float64)
    rates = 100 * conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=100)
    for r in range(len(ATR_CLASSES)):
        for c in range(len(ATR_CLASSES)):
            if rates[r, c] >= 1:
                ax.text(c, r, f"{rates[r, c]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if rates[r, c] > 50 else "black")
    ax.set_xticks(range(len(ATR_CLASSES)), ATR_CLASSES, rotation=90, fontsize="small")
    ax.set_yticks(range(len(ATR_CLASSES)), ATR_CLASSES, fontsize="small")
    ax.set(xlabel="predicted class", ylabel="true class",
           title="Pixels of each true class, by predicted class (%)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training history and final result
# ---------------------------------------------------------------------------

class SegmentationHistory:
    """Record the training loss, validation results and learning rate of each epoch.

    ``log`` takes the average training loss and the validation metrics from
    ``evaluate_segmentation``. The best epoch is the one with the highest
    validation mIoU.
    """

    def __init__(self, name="run", config=None, tensorboard=True, logdir="runs"):
        self.name, self.config = name, dict(config or {})
        self.epoch, self.lr, self.train_loss, self.val = [], [], [], []
        self.seconds = []
        self.writer = None
        self._start = time.time()
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"{logdir}/{name}_{time.strftime('%Y%m%d-%H%M%S')}")

    def log(self, epoch, train_loss, val_metrics, lr=None, verbose=True):
        self.epoch.append(epoch)
        self.train_loss.append(float(train_loss))
        self.val.append({k: v for k, v in val_metrics.items() if k != "class_iou"})
        self.lr.append(lr)
        self.seconds.append(time.time() - self._start)
        if verbose:
            lr_text = f"  lr {lr:.2e}" if lr is not None else ""
            best = "  *best" if self.is_best() else ""
            print(f"Epoch {epoch + 1:3d} | train loss {train_loss:.4f} | val loss {val_metrics['loss']:.4f}  "
                  f"pixel acc {val_metrics['pixel_acc']:5.1f}%  mIoU {val_metrics['mIoU']:5.1f}%"
                  f"{lr_text}  ({self.seconds[-1]:.0f} s){best}")
        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/validation", val_metrics["loss"], epoch)
            self.writer.add_scalar("Validation/mIoU", val_metrics["mIoU"], epoch)
            self.writer.add_scalar("Validation/pixel accuracy", val_metrics["pixel_acc"], epoch)
            if lr is not None:
                self.writer.add_scalar("Learning rate", lr, epoch)
            self.writer.flush()

    def best_index(self):
        """Index of the epoch with the highest validation mIoU (the first one wins a tie)."""
        return int(np.argmax([v["mIoU"] for v in self.val]))

    def is_best(self):
        """True if the epoch logged last is the best so far (use it to keep the best weights)."""
        return bool(self.epoch) and self.best_index() == len(self.epoch) - 1

    def plot(self):
        """Plot the losses, validation mIoU and pixel accuracy, and the learning rate."""
        if not self.epoch:
            print("Nothing to plot: no epochs have been logged yet.")
            return
        epochs = [e + 1 for e in self.epoch]
        has_lr = any(lr is not None for lr in self.lr)
        fig, axes = plt.subplots(1, 3 if has_lr else 2, figsize=(15 if has_lr else 11, 4))
        axes[0].plot(epochs, self.train_loss, label="train")
        axes[0].plot(epochs, [v["loss"] for v in self.val], label="validation")
        axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")
        axes[1].plot(epochs, [v["mIoU"] for v in self.val], label="mIoU")
        axes[1].plot(epochs, [v["pixel_acc"] for v in self.val], label="pixel accuracy")
        axes[1].set(title="Validation", xlabel="epoch", ylabel="%")
        if has_lr:
            axes[2].plot(epochs, self.lr, color="tab:green")
            axes[2].set(title="Learning rate", xlabel="epoch", yscale="log")
        for ax in axes:
            ax.grid(alpha=0.3)
        for ax in axes[:2]:
            ax.legend()
        fig.suptitle(self.name)
        fig.tight_layout()
        plt.show()

    def summary(self):
        """Print (and return as a dict) the settings and best validation results of this run."""
        if not self.epoch:
            print("No epochs logged yet.")
            return {}
        best = self.best_index()
        result = {"run": self.name,
                  **{k: v for k, v in self.config.items() if k != "run_name"},
                  "epochs trained": len(self.epoch), "best epoch": self.epoch[best] + 1,
                  "best val mIoU": round(self.val[best]["mIoU"], 1),
                  "best val pixel acc": round(self.val[best]["pixel_acc"], 1),
                  "training time (min)": round(self.seconds[-1] / 60, 1)}
        width = max(len(k) for k in result)
        for key, value in result.items():
            print(f"{key:<{width}} : {value}")
        return result


@torch.no_grad()
def final_segmentation_result(model, history, test_loader, device):
    """Evaluate the network once on the test set and print the result block.

    Returns a dict with all values shown in the block (with the IoU of every class).
    """
    from .models import compute_cost, format_count

    preds, masks = predict_segmentation(model, test_loader, device)
    metrics = metrics_from_confusion(confusion_matrix(preds, masks))
    image_size = tuple(masks.shape[1:])
    cost = compute_cost(model, image_size, device, verbose=False)
    best = history.best_index() if history.epoch else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
    result = {
        "run": history.name,
        "image size": f"{image_size[0]} x {image_size[1]}",
        "test images": len(masks),
        "epochs trained": len(history.epoch),
        "weights from epoch": history.epoch[best] + 1 if history.epoch else None,
        "val mIoU (%)": round(history.val[best]["mIoU"], 1) if history.epoch else None,
        "TEST mIoU (%)": round(metrics["mIoU"], 1),
        "test pixel acc (%)": round(metrics["pixel_acc"], 1),
        "parameters": format_count(cost["params"]),
        "inference GFLOP per image": round(cost["inference_flops"] / 1e9, 2),
        "training time (min)": round(history.seconds[-1] / 60, 1) if history.epoch else None,
        "settings": history.config,
        "evaluated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                     + f" on {gpu}",
    }
    width = max(len(k) for k in result)
    line = "=" * 72
    print(f"{line}\nFINAL RESULT (test set, evaluated once)\n{line}")
    for key, value in result.items():
        print(f"{key:<{width}} : {value}")
    print("IoU of each class (%):")
    print("  " + ", ".join(f"{name} {iou:.0f}" for name, iou in metrics["class_iou"].items()))
    print(line)
    result["class IoU (%)"] = {k: round(v, 1) for k, v in metrics["class_iou"].items()}
    return result
