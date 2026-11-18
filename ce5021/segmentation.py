"""Semantic segmentation: the Oxford-IIIT Pet dataset (background, cat, dog), pre-trained
encoder stages, evaluation (IoU), the challenge score and plots.

The network and the training loop are written in the notebook::

    train_loader, val_loader, test_loader = pet_loaders(image_size, train_tf, test_tf)
    logits = model(images)                    # [B, 3, H, W], same H, W as the images
    loss = loss_fn(logits, masks)             # masks: [B, H, W] class index of every pixel

    metrics = evaluate_segmentation(model, val_loader, loss_fn, device)   # loss, mIoU, pixel acc
    preds, masks = predict_segmentation(model, val_loader, device)
    show_segmentation(val_loader.dataset, preds, indices=worst_images(preds, masks))

The pixels on the outline of each animal are marked ``IGNORE_INDEX`` (255): they
are left out of the loss (``nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)``)
and of every metric.

Validation and test results are always measured at ``EVAL_SIZE`` (256 x 256):
the network sees images of the chosen ``image_size``, and its class scores are
upsampled to 256 x 256 before they are compared with the masks.
"""

import datetime
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import _download, denormalize

# The Oxford-IIIT Pet dataset (Parkhi et al., "Cats and Dogs", CVPR 2012): 7,349
# photographs of 37 breeds of cats and dogs, each with a mask ("trimap") of the
# animal, its outline and the background. The breeds are grouped into cat and dog.
PET_URL = ("https://huggingface.co/datasets/dpdl-benchmark/oxford_iiit_pet/resolve/main/data/"
           "{split}-0000{i}-of-00003.parquet")
PET_FILES = [("train", i) for i in range(3)] + [("test", i) for i in range(3)]
PET_TRAINVAL, PET_TEST = 3_680, 3_669      # the official split of the dataset
PET_SIZE = PET_TRAINVAL + PET_TEST
PET_VAL = 736                              # validation images, taken from the 3,680 at random
PET_CLASSES = ["background", "cat", "dog"]
IGNORE_INDEX = 255                         # outline pixels: not used in the loss or the metrics
EVAL_SIZE = 256                            # images are stored, and results measured, at 256 x 256

# Colour of each class in the plots (index = class; IGNORE_INDEX light grey).
CLASS_COLOURS = np.zeros((256, 3), dtype=np.uint8)
CLASS_COLOURS[:3] = [(35, 35, 35), (235, 104, 52), (42, 120, 214)]
CLASS_COLOURS[IGNORE_INDEX] = (205, 205, 205)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def download_pet(root="data"):
    """Download the Oxford-IIIT Pet parquet files from Hugging Face (first time only, 2.3 GB).

    Returns the list of downloaded files: the 3,680 training and validation
    images first, then the 3,669 test images.
    """
    folder = os.path.join(root, "pet")
    os.makedirs(folder, exist_ok=True)
    paths = []
    for n, (split, i) in enumerate(PET_FILES):
        path = os.path.join(folder, f"{split}-{i}.parquet")
        if not os.path.exists(path):
            print(f"Downloading Oxford-IIIT Pet part {n + 1} of {len(PET_FILES)} (about 380 MB) ...")
            _download(PET_URL.format(split=split, i=i), path)
        paths.append(path)
    return paths


def pet_mask(trimap, species):
    """Trimap (1 animal, 2 background, 3 outline) -> classes (0 background, 1 cat, 2 dog, 255 outline).

    ``species`` is 0 for a cat and 1 for a dog.
    """
    trimap = np.asarray(trimap)
    return np.where(trimap == 1, 1 + species,
                    np.where(trimap == 2, 0, IGNORE_INDEX)).astype(np.uint8)


def prepare_pet(root="data"):
    """Resize every image and mask to 256 x 256 once and store them as arrays.

    Returns the paths of ``images`` [N, 256, 256, 3] and ``masks`` [N, 256, 256]
    (uint8 ``.npy`` files, read from disk while training rather than held in
    memory). Images are resized smoothly; masks with the nearest pixel, so every
    pixel keeps a valid class. The photographs have different shapes, so they
    are stretched to a square.
    """
    import pyarrow.parquet as pq     # preinstalled in Colab
    from PIL import Image

    folder = os.path.join(root, "pet")
    images_path = os.path.join(folder, f"images_{EVAL_SIZE}.npy")
    masks_path = os.path.join(folder, f"masks_{EVAL_SIZE}.npy")
    if os.path.exists(images_path) and os.path.exists(masks_path):
        return images_path, masks_path

    parquet_files = download_pet(root)
    print(f"Resizing the images to {EVAL_SIZE} x {EVAL_SIZE} (first time only, about two minutes) ...")
    size = (EVAL_SIZE, EVAL_SIZE)
    images = np.lib.format.open_memmap(images_path + ".part", mode="w+", dtype=np.uint8,
                                       shape=(PET_SIZE, *size, 3))
    masks = np.lib.format.open_memmap(masks_path + ".part", mode="w+", dtype=np.uint8,
                                      shape=(PET_SIZE, *size))
    n = 0
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):      # one row group at a time: low memory
            for row in parquet.read_row_group(group).to_pylist():
                image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                trimap = Image.open(io.BytesIO(row["segmentation_mask"]["bytes"])).convert("L")
                images[n] = np.asarray(image.resize(size, Image.BILINEAR))
                masks[n] = pet_mask(trimap.resize(size, Image.NEAREST), row["species"])
                n += 1
    if n != PET_SIZE:
        raise ValueError(f"unexpected contents in {folder}: {n} images instead of {PET_SIZE}")
    images.flush(), masks.flush()
    del images, masks
    os.replace(images_path + ".part", images_path)
    os.replace(masks_path + ".part", masks_path)
    return images_path, masks_path


def _as_size(image_size):
    """64 -> (64, 64); (h, w) unchanged. Checked: multiples of 32, at most EVAL_SIZE."""
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    if len(size) != 2 or any(s % 32 or not 32 <= s <= EVAL_SIZE for s in size):
        raise ValueError(f"image_size must be a multiple of 32 between 32 and {EVAL_SIZE} "
                         f"(or a (height, width) pair of them), not {image_size}")
    return size


class PetSegmentation(Dataset):
    """Pet images and masks (stored at 256 x 256), selected by ``indices``.

    Each item is ``(image, mask)``. The image is resized to ``image_size`` and
    then passed through ``transform`` (a torchvision ``v2`` transform applied to
    the image and the mask together: geometric transforms move the mask with the
    image; colour transforms only change the image). The mask is a tensor
    [H, W] of class indices (int64), resized to ``image_size`` too when
    ``mask_at_image_size`` (training), or kept at 256 x 256 (validation, test).
    """

    def __init__(self, images_path, masks_path, indices, image_size=EVAL_SIZE, transform=None,
                 mask_at_image_size=True):
        self.images_path, self.masks_path = images_path, masks_path
        self.indices, self.transform = np.asarray(indices), transform
        self.image_size, self.mask_at_image_size = _as_size(image_size), mask_at_image_size
        self._images = self._masks = None
        self.classes = PET_CLASSES

    def __len__(self):
        return len(self.indices)

    def class_pixel_counts(self):
        """Pixels of each class (and of the ignored outline) over all masks of this dataset, at 256 x 256.

        Returns ``(counts, outline, images_with_class)``.
        """
        masks = np.load(self.masks_path, mmap_mode="r")
        counts = np.zeros(256, dtype=np.int64)
        images_with = np.zeros(len(PET_CLASSES), dtype=np.int64)
        for start in range(0, len(self.indices), 500):            # in chunks, to keep memory low
            chunk = masks[np.sort(self.indices[start:start + 500])]
            counts += np.bincount(chunk.reshape(-1), minlength=256)
            for c in range(len(PET_CLASSES)):
                images_with[c] += int((chunk == c).any(axis=(1, 2)).sum())
        return counts[:len(PET_CLASSES)], int(counts[IGNORE_INDEX]), images_with

    def __getitem__(self, index):
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as TF
        if self._images is None:          # opened here, so each data-loading worker opens its own
            self._images = np.load(self.images_path, mmap_mode="r")
            self._masks = np.load(self.masks_path, mmap_mode="r")
        i = self.indices[index]
        image = tv_tensors.Image(torch.from_numpy(np.array(self._images[i])).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(np.array(self._masks[i])))
        if self.image_size != (EVAL_SIZE, EVAL_SIZE):
            image = TF.resize(image, list(self.image_size), antialias=True)
            if self.mask_at_image_size:
                mask = TF.resize(mask, list(self.image_size))      # nearest pixel for a mask
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, torch.as_tensor(mask).long()


def pet_loaders(image_size, train_transform, test_transform, batch_size=32, n_train=None,
                root="data", seed=0, num_workers=2):
    """Download the Oxford-IIIT Pet dataset and return (train_loader, val_loader, test_loader).

    Args:
        image_size: size of the images given to the network: a multiple of 32
            up to 256, or a (height, width) pair of them.
        n_train: number of training images to use (None: all 2,944), for
            shorter experiments. The validation and test sets are always complete.

    The 3,680 images of the official training split are divided at random
    (fixed by ``seed``) into 2,944 training and 736 validation images; the
    3,669 images of the official test split are the test set. A batch is
    ``(images, masks)``: images [B, 3, H, W] and masks [B, H, W] holding the
    class of every pixel (0 background, 1 cat, 2 dog, 255 outline). Training
    masks have the size of the images; validation and test masks are always
    256 x 256, so that every design is measured at the same resolution.
    """
    image_size = _as_size(image_size)
    images_path, masks_path = prepare_pet(root)
    order = np.random.default_rng(seed).permutation(PET_TRAINVAL)
    parts = [order[PET_VAL:][:n_train], np.sort(order[:PET_VAL]), np.arange(PET_TRAINVAL, PET_SIZE)]
    datasets = [PetSegmentation(images_path, masks_path, idx, image_size, tf, mask_at_image_size=train)
                for idx, tf, train in zip(parts, (train_transform, test_transform, test_transform),
                                          (True, False, False))]
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(datasets[0], shuffle=True, drop_last=True, **kwargs),
            DataLoader(datasets[1], **kwargs), DataLoader(datasets[2], **kwargs))


# ---------------------------------------------------------------------------
# Pre-trained encoders
# ---------------------------------------------------------------------------

# For each network: the slices of its layers that form the 5 stages (1/2 ... 1/32 of the input).
_ENCODER_STAGES = {
    "resnet18": None, "resnet34": None, "resnet50": None,
    "mobilenet_v3_large": ((0, 2), (2, 4), (4, 7), (7, 13), (13, 16)),
    "efficientnet_b0": ((0, 2), (2, 3), (3, 4), (4, 6), (6, 8)),
}
ENCODERS = tuple(_ENCODER_STAGES)


def encoder_stages(arch="resnet18", pretrained=True):
    """The stages of an ImageNet-trained classification network, as separate modules, for an encoder.

    ``arch`` is one of ``ENCODERS``. Returns ``(stages, channels)``: ``stages``
    is an ``nn.ModuleList`` of 5 modules applied one after the other; the
    output of stage k has ``channels[k]`` channels and 1/2, 1/4, 1/8, 1/16,
    1/32 of the input height and width::

        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)     # kept for the skip connections of a decoder

    The classifier (average pooling and fully connected layers) is not included.
    The stages are trained with the rest of the network unless their parameters
    are frozen (``requires_grad = False``).
    """
    from torchvision import models as tv_models
    if arch not in ENCODERS:
        raise ValueError(f"arch must be one of {ENCODERS}, not {arch!r}")
    net = getattr(tv_models, arch)(weights="DEFAULT" if pretrained else None)
    if arch.startswith("resnet"):
        stages = nn.ModuleList([
            nn.Sequential(net.conv1, net.bn1, net.relu),     # 1/2
            nn.Sequential(net.maxpool, net.layer1),          # 1/4
            net.layer2,                                      # 1/8
            net.layer3,                                      # 1/16
            net.layer4,                                      # 1/32
        ])
    else:
        stages = nn.ModuleList([net.features[a:b] for a, b in _ENCODER_STAGES[arch]])
    channels, x = [], torch.zeros(1, 3, 64, 64)
    was_training = stages.training
    stages.eval()
    with torch.no_grad():
        for stage in stages:
            x = stage(x)
            channels.append(x.shape[1])
    stages.train(was_training)
    return stages, channels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def confusion_matrix(preds, masks, n_classes=len(PET_CLASSES)):
    """Pixel counts [n_classes, n_classes]: row = true class, column = predicted class.

    Pixels marked ``IGNORE_INDEX`` in ``masks`` are left out. The pixels are
    counted in chunks, so that the whole test set fits in memory.
    """
    preds = (preds.numpy() if torch.is_tensor(preds) else np.asarray(preds)).reshape(-1)
    masks = (masks.numpy() if torch.is_tensor(masks) else np.asarray(masks)).reshape(-1)
    counts = np.zeros(n_classes ** 2, dtype=np.int64)
    chunk = 1 << 24
    for start in range(0, len(masks), chunk):
        m = masks[start:start + chunk].astype(np.int64)
        p = preds[start:start + chunk].astype(np.int64)
        keep = m < n_classes
        counts += np.bincount(m[keep] * n_classes + p[keep], minlength=n_classes ** 2)
    return counts.reshape(n_classes, n_classes)


def metrics_from_confusion(conf):
    """Pixel accuracy, mean IoU and the IoU of each class (all in %) from a confusion matrix.

    The IoU of a class is (pixels correctly given the class) / (pixels that
    are the class or are predicted as the class). The mean IoU (mIoU) is the
    average over the classes, so the cats count as much as the background.
    """
    conf = conf.astype(np.float64)
    hits = np.diag(conf)
    union = conf.sum(axis=0) + conf.sum(axis=1) - hits
    iou = np.where(union > 0, hits / np.maximum(union, 1), np.nan)
    return {"pixel_acc": 100 * hits.sum() / max(conf.sum(), 1),
            "mIoU": 100 * float(np.nanmean(iou)),
            "class_iou": {name: 100 * float(v) for name, v in zip(PET_CLASSES, iou)}}


def _check_output(logits, images):
    if logits.shape[-2:] != images.shape[-2:]:
        raise ValueError(f"the network output {tuple(logits.shape)} must have the same height and "
                         f"width as the images {tuple(images.shape)}")


def _scores_at_mask_size(model, images, masks):
    """Logits of the model, upsampled (bilinear) to the size of the masks if they differ."""
    logits = model(images)
    _check_output(logits, images)
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits.float(), size=masks.shape[-2:], mode="bilinear",
                               align_corners=False)
    return logits


@torch.no_grad()
def evaluate_segmentation(model, loader, loss_fn, device):
    """Average loss, pixel accuracy, mIoU and class IoU (in %) of the model on ``loader``.

    The class scores are upsampled to the size of the masks (256 x 256 for the
    validation and test loaders) before the loss and the metrics are computed.
    """
    model.eval()
    total_loss, seen = 0.0, 0
    conf = np.zeros((len(PET_CLASSES), len(PET_CLASSES)), dtype=np.int64)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = _scores_at_mask_size(model, images, masks)
        total_loss += loss_fn(logits, masks).item() * len(masks)
        seen += len(masks)
        conf += confusion_matrix(logits.argmax(dim=1).cpu(), masks.cpu())
    return {"loss": total_loss / seen, **metrics_from_confusion(conf)}


@torch.no_grad()
def predict_segmentation(model, loader, device):
    """Return (preds, masks): predicted and true class of every pixel, uint8 arrays [N, H, W].

    The predictions have the size of the masks (256 x 256 for the validation
    and test loaders).
    """
    model.eval()
    preds, masks = [], []
    for images, true in loader:
        logits = _scores_at_mask_size(model, images.to(device), true)
        preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
        masks.append(true.numpy().astype(np.uint8))
    return np.concatenate(preds), np.concatenate(masks)


def image_miou(pred, mask):
    """Mean IoU (%) of one image, over the classes that appear in its mask or prediction.

    Outline pixels (``IGNORE_INDEX``) are left out.
    """
    pred, mask = np.asarray(pred), np.asarray(mask)
    keep = mask != IGNORE_INDEX
    pred, mask = pred[keep], mask[keep]
    ious = []
    for c in np.union1d(np.unique(pred), np.unique(mask)):
        p, m = pred == c, mask == c
        ious.append((p & m).sum() / (p | m).sum())
    return 100 * float(np.mean(ious)) if ious else float("nan")


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
    labels = PET_CLASSES + ["outline (ignored)"]
    colours = [CLASS_COLOURS[i] for i in range(len(PET_CLASSES))] + [CLASS_COLOURS[IGNORE_INDEX]]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c / 255, edgecolor="gray") for c in colours]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize="small", frameon=False)


def show_segmentation(dataset, preds=None, indices=None, n=4, title=""):
    """Show images with their true masks (and predicted masks, if ``preds`` is given).

    ``preds`` comes from ``predict_segmentation`` with a loader over ``dataset``
    that does not shuffle (validation or test), so it is in the same order as
    the dataset. The mean IoU of each image is shown above its prediction.
    """
    indices = list(range(min(n, len(dataset)))) if indices is None else list(indices)[:n]
    rows = 3 if preds is not None else 2
    fig, axes = plt.subplots(rows, len(indices), figsize=(2.6 * len(indices), 2.95 * rows + 0.7),
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
    """Bar chart of the share of pixels of each class in ``dataset``, with the number of images.

    The outline pixels, which are not used, are shown separately.
    """
    counts, outline, images_with = dataset.class_pixel_counts()
    total = counts.sum() + outline
    names = PET_CLASSES + ["outline\n(ignored)"]
    share = 100 * np.append(counts, outline) / total
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colours = [CLASS_COLOURS[i] / 255 for i in range(len(PET_CLASSES))] + [CLASS_COLOURS[IGNORE_INDEX] / 255]
    bars = ax.bar(names, share, color=colours, edgecolor="gray")
    ax.bar_label(bars, labels=[f"{s:.1f}%" for s in share], fontsize="small")
    ax.set(ylabel="share of pixels (%)", ylim=(0, 100 * 1.12 * share.max() / 100),
           title=f"{len(dataset)} images: {images_with[1]} with a cat, {images_with[2]} with a dog")
    fig.tight_layout()
    plt.show()
    return dict(zip(PET_CLASSES + ["outline"], share))


def plot_class_iou(metrics, title="IoU of each class"):
    """Bar chart of the IoU of each class, with the mean IoU as a line."""
    names, values = list(metrics["class_iou"]), list(metrics["class_iou"].values())
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(names, values, color=CLASS_COLOURS[np.arange(len(names))] / 255, edgecolor="gray")
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize="small")
    ax.axhline(metrics["mIoU"], color="tab:red", linestyle="--", label=f"mIoU {metrics['mIoU']:.1f}%")
    ax.set(ylim=(0, 105), ylabel="IoU (%)", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def plot_confusion(preds, masks):
    """Confusion matrix of the pixels, each row divided by its total (recall of each class).

    Values off the diagonal show which classes are mistaken for which: cat and
    dog pixels predicted as background are missed parts of the animal; cat
    pixels predicted as dog are a wrong recognition of the animal.
    """
    conf = confusion_matrix(preds, masks).astype(np.float64)
    rates = 100 * conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=100)
    for r in range(len(PET_CLASSES)):
        for c in range(len(PET_CLASSES)):
            ax.text(c, r, f"{rates[r, c]:.1f}", ha="center", va="center",
                    color="white" if rates[r, c] > 50 else "black")
    ax.set_xticks(range(len(PET_CLASSES)), PET_CLASSES)
    ax.set_yticks(range(len(PET_CLASSES)), PET_CLASSES)
    ax.set(xlabel="predicted class", ylabel="true class", title="Pixels of each true class (%)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training history, score and final result
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


# The challenge score: S = test mIoU (%) x (training compute / SCORE_REFERENCE_TFLOP) ** SCORE_EXPONENT
SCORE_REFERENCE_TFLOP = 10
SCORE_EXPONENT = -0.035


def segmentation_score(miou, compute_tflop, reference_tflop=SCORE_REFERENCE_TFLOP,
                       exponent=SCORE_EXPONENT):
    """Score of the segmentation challenge (Assignment 3).

        S = test mIoU (%) x (training compute / 10 TFLOP) ** -0.035

    As in Assignment 2, compute is a soft penalty: doubling the training
    compute lowers the score by about 2.4 %, and ten times the compute by
    about 8 %. The exponent is smaller than in Assignment 2 because the mIoU
    of good designs rises more slowly with compute.
    """
    return miou * (compute_tflop / reference_tflop) ** exponent


def _size_text(size):
    """(64, 64) -> 64; (64, 48) -> "64 x 48"."""
    return size[0] if size[0] == size[1] else f"{size[0]} x {size[1]}"


@torch.no_grad()
def final_segmentation_result(model, history, train_loader, test_loader, device):
    """Evaluate the network once on the test set and print the result block with the score S.

    The training compute is recalculated here from the network itself (which
    layers are trained), the size of the training images as they are loaded
    (after the transforms), the number of training images per epoch and the
    number of epochs in ``history``:

        training compute = training FLOPs per image x images per epoch x epochs

    Returns a dict with all values shown in the block (with the IoU of every class).
    """
    from .models import compute_cost, format_count

    preds, masks = predict_segmentation(model, test_loader, device)
    metrics = metrics_from_confusion(confusion_matrix(preds, masks))
    train_size = tuple(next(iter(train_loader))[0].shape[-2:])
    test_size = tuple(next(iter(test_loader))[0].shape[-2:])
    cost = compute_cost(model, train_size, device, verbose=False)
    images_per_epoch = len(train_loader.sampler)
    epochs = len(history.epoch)
    compute = cost["train_flops"] * images_per_epoch * epochs
    best = history.best_index() if epochs else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
    result = {
        "run": history.name,
        "image size (network input)": _size_text(train_size),
        **({"test image size": _size_text(test_size)} if test_size != train_size else {}),
        "masks scored at": _size_text(tuple(masks.shape[1:])),
        "test images": len(masks),
        "training images per epoch": images_per_epoch,
        "epochs trained": epochs,
        "weights from epoch": history.epoch[best] + 1 if epochs else None,
        "val mIoU (%)": round(history.val[best]["mIoU"], 1) if epochs else None,
        "TEST mIoU (%)": round(metrics["mIoU"], 1),
        "test pixel acc (%)": round(metrics["pixel_acc"], 1),
        "parameters": format_count(cost["params"]),
        "trained parameters": format_count(cost["trainable_params"]),
        "inference GFLOP per image": round(cost["inference_flops"] / 1e9, 3),
        "training GFLOP per image": round(cost["train_flops"] / 1e9, 3),
        "TRAINING COMPUTE (TFLOP)": round(compute / 1e12, 2),
        "SCORE S": round(segmentation_score(metrics["mIoU"], compute / 1e12), 1) if compute else None,
        "training time (min)": round(history.seconds[-1] / 60, 1) if epochs else None,
        "settings": history.config,
        "evaluated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                     + f" on {gpu}",
    }
    width = max(len(k) for k in result)
    line = "=" * 72
    print(f"{line}\nFINAL RESULT (test set, evaluated once)\n{line}")
    for key, value in result.items():
        print(f"{key:<{width}} : {value}")
    print("IoU of each class (%):  " + ", ".join(f"{name} {iou:.1f}"
                                                  for name, iou in metrics["class_iou"].items()))
    print(line)
    if test_size != train_size:
        print(f"WARNING: the training images ({_size_text(train_size)}) and test images "
              f"({_size_text(test_size)}) differ in size.\nThe training compute is calculated "
              "from the size of the training images. Check the size given to the transforms.")
    result["class IoU (%)"] = {k: round(v, 1) for k, v in metrics["class_iou"].items()}
    return result
