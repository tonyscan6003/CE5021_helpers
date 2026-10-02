"""Object detection: the Penn-Fudan dataset, backbones, evaluation (AP) and debugging plots.

The detector itself (its head, target encoding, loss and decoding) and the
training loop are written in the notebook. These helpers are the parts that
stay the same for every single-stage design::

    train_loader, val_loader, test_loader = pennfudan_loaders(image_size, train_tf, test_tf)
    backbone, channels, stride = build_backbone("resnet18", "layer4")

    output = model(images)                    # any tensor (or dict of tensors)
    detections = decode(output)               # list of {"boxes": [N, 4], "scores": [N]}

    preds, gts = predict_detections(model, val_loader, decode, device)
    metrics = detection_metrics(preds, gts)   # AP, AP50, AP75, recall

Boxes are always ``[x1, y1, x2, y2]`` in pixels of the (letterboxed) input image.
"""

import datetime
import os
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import IMAGENET_MEAN, IMAGENET_STD, denormalize

PENNFUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
PENNFUDAN_SPLIT = (100, 20, 50)      # training, validation, test images (170 in total)
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)   # COCO: 0.50, 0.55, ..., 0.95


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def download_pennfudan(root="data"):
    """Download and unpack the Penn-Fudan pedestrian dataset (first time only).

    Returns the dataset folder (``<root>/PennFudanPed``).
    """
    folder = os.path.join(root, "PennFudanPed")
    if not os.path.isdir(os.path.join(folder, "PNGImages")):
        import urllib.request
        os.makedirs(root, exist_ok=True)
        archive = os.path.join(root, "PennFudanPed.zip")
        if not os.path.exists(archive):
            print("Downloading Penn-Fudan (about 50 MB) ...")
            urllib.request.urlretrieve(PENNFUDAN_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    return folder


def _boxes_from_mask(mask_path):
    """One box per pedestrian from an instance mask (pixel value i = pedestrian i)."""
    from PIL import Image
    mask = np.array(Image.open(mask_path))
    boxes = []
    for i in np.unique(mask):
        if i == 0:                                    # 0 = background
            continue
        ys, xs = np.nonzero(mask == i)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def letterbox(image, boxes, size):
    """Scale an image so its longer side is ``size`` and pad it to ``size`` x ``size``.

    The image keeps its aspect ratio (pedestrians are not stretched); the
    padding is added on the right or at the bottom, so box coordinates only
    need to be scaled. Returns (padded PIL image, scaled boxes).
    """
    from PIL import Image
    scale = size / max(image.size)
    new_w, new_h = round(image.width * scale), round(image.height * scale)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))      # ImageNet mean colour
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (0, 0))
    return canvas, boxes * scale


class PennFudanDetection(Dataset):
    """Penn-Fudan images (letterboxed to ``size`` x ``size``) with their pedestrian boxes.

    Each item is ``(image tensor [3, size, size], boxes tensor [N, 4])``.
    ``transform`` is a torchvision ``v2`` transform; geometric transforms
    (flips, crops, affine) move the boxes with the image. Boxes that end up
    (almost) outside the image are dropped.
    """

    def __init__(self, folder, names, size, transform=None):
        self.folder, self.names, self.size, self.transform = folder, list(names), size, transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        from PIL import Image
        from torchvision import tv_tensors

        name = self.names[index]
        image = Image.open(os.path.join(self.folder, "PNGImages", name)).convert("RGB")
        boxes = _boxes_from_mask(os.path.join(self.folder, "PedMasks",
                                              name.replace(".png", "_mask.png")))
        image, boxes = letterbox(image, boxes, self.size)
        boxes = tv_tensors.BoundingBoxes(torch.as_tensor(boxes), format="XYXY",
                                         canvas_size=(self.size, self.size))
        if self.transform is not None:
            image, boxes = self.transform(image, boxes)
        h, w = image.shape[-2:]
        boxes = torch.as_tensor(boxes).float()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        keep = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
        return image, boxes[keep]


def collate_detection(batch):
    """Stack the images of a batch; keep the boxes as a list (each image has a different number)."""
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


def pennfudan_loaders(image_size, train_transform, test_transform, batch_size=8,
                      root="data", split=PENNFUDAN_SPLIT, seed=0, num_workers=2):
    """Download Penn-Fudan and return (train_loader, val_loader, test_loader).

    The 170 images are split at random (fixed by ``seed``) into ``split``
    training, validation and test images. Every image is letterboxed to
    ``image_size`` x ``image_size`` pixels. A batch is ``(images, boxes)``:
    ``images`` is a tensor [B, 3, image_size, image_size] and ``boxes`` a list
    of B tensors [N, 4] (``x1, y1, x2, y2`` in pixels).
    """
    if image_size % 32:
        raise ValueError(f"image_size must be a multiple of 32, not {image_size}")
    folder = download_pennfudan(root)
    names = sorted(os.listdir(os.path.join(folder, "PNGImages")))
    order = np.random.default_rng(seed).permutation(len(names))
    n_train, n_val, n_test = split
    parts = [order[:n_train], order[n_train:n_train + n_val],
             order[n_train + n_val:n_train + n_val + n_test]]
    datasets = [PennFudanDetection(folder, [names[i] for i in idx], image_size, tf)
                for idx, tf in zip(parts, (train_transform, test_transform, test_transform))]
    kwargs = dict(batch_size=batch_size, collate_fn=collate_detection, num_workers=num_workers)
    return (DataLoader(datasets[0], shuffle=True, drop_last=True, **kwargs),
            DataLoader(datasets[1], **kwargs), DataLoader(datasets[2], **kwargs))


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

BACKBONES = ("resnet18", "resnet34", "resnet50")
_STAGES = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}


def build_backbone(arch="resnet18", stop_at="layer4", pretrained=True):
    """An ImageNet-trained ResNet, cut after ``stop_at``, as a feature extractor.

    Returns ``(backbone, channels, stride)``: ``backbone`` maps images
    [B, 3, H, W] to feature maps [B, channels, H / stride, W / stride].

        stop_at   stride   resnet18/34 channels   resnet50 channels
        layer1       4             64                   256
        layer2       8            128                   512
        layer3      16            256                  1024
        layer4      32            512                  2048

    The classifier (average pooling and the fully connected layer) is removed,
    so the spatial layout of the feature maps is kept.
    """
    from torchvision import models as tv_models
    if arch not in BACKBONES:
        raise ValueError(f"arch must be one of {BACKBONES}, not {arch!r}")
    if stop_at not in _STAGES:
        raise ValueError(f"stop_at must be one of {list(_STAGES)}, not {stop_at!r}")
    net = getattr(tv_models, arch)(weights="DEFAULT" if pretrained else None)
    layers = [net.conv1, net.bn1, net.relu, net.maxpool]
    for stage in _STAGES:
        layers.append(getattr(net, stage))
        if stage == stop_at:
            break
    backbone = nn.Sequential(*layers)
    last = layers[-1][-1]
    channels = (last.conv3 if hasattr(last, "conv3") else last.conv2).out_channels
    return backbone, channels, _STAGES[stop_at]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def box_iou(a, b):
    """Intersection over union of every box in ``a`` [N, 4] with every box in ``b`` [M, 4]."""
    from torchvision.ops import box_iou as tv_box_iou
    return tv_box_iou(torch.as_tensor(a, dtype=torch.float32).reshape(-1, 4),
                      torch.as_tensor(b, dtype=torch.float32).reshape(-1, 4))


@torch.no_grad()
def predict_detections(model, loader, decode, device, **decode_kwargs):
    """Run the model and ``decode`` over a loader.

    Returns ``(preds, gts)``: ``preds`` is a list (one per image) of dicts
    with ``boxes`` [N, 4] and ``scores`` [N]; ``gts`` a list of the true boxes.
    Extra keyword arguments are passed to ``decode`` (e.g. ``nms_iou=None``).
    """
    model.eval()
    preds, gts = [], []
    for images, boxes in loader:
        detections = decode(model(images.to(device)), **decode_kwargs)
        preds += [{k: v.detach().float().cpu() for k, v in d.items()} for d in detections]
        gts += [b.cpu() for b in boxes]
    return preds, gts


def _match(pred, gt, iou_threshold):
    """Greedy COCO matching for one image: is each prediction (in score order) a true positive?"""
    order = torch.argsort(pred["scores"], descending=True)
    scores = pred["scores"][order].numpy()
    tp = np.zeros(len(order), dtype=bool)
    if len(gt) and len(order):
        ious = box_iou(pred["boxes"][order], gt).numpy()
        taken = np.zeros(len(gt), dtype=bool)
        for k in range(len(order)):
            candidates = np.where(taken, -1.0, ious[k])
            j = int(candidates.argmax())
            if candidates[j] >= iou_threshold:
                tp[k], taken[j] = True, True
    return scores, tp


def average_precision(preds, gts, iou_threshold=0.5):
    """Average precision (area under the precision-recall curve) at one IoU threshold.

    COCO definition: detections are matched to the ground truth in order of
    score, each true box at most once; precision is interpolated at 101
    recall levels. Returns (AP, recall at the lowest score threshold).
    """
    n_gt = sum(len(g) for g in gts)
    scores, tps = [], []
    for pred, gt in zip(preds, gts):
        s, tp = _match(pred, gt, iou_threshold)
        scores.append(s)
        tps.append(tp)
    scores, tps = np.concatenate(scores), np.concatenate(tps)
    if n_gt == 0 or len(scores) == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="stable")
    tp_cum = np.cumsum(tps[order])
    fp_cum = np.cumsum(~tps[order])
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)
    precision = np.maximum.accumulate(precision[::-1])[::-1]     # precision envelope
    levels = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, levels, side="left")
    interpolated = np.where(idx < len(precision),
                            precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(interpolated.mean()), float(recall[-1])


def detection_metrics(preds, gts):
    """AP (COCO: mean over IoU 0.50-0.95), AP50, AP75 and recall at IoU 0.5, in %."""
    aps = {t: average_precision(preds, gts, t) for t in IOU_THRESHOLDS}
    return {"AP": 100 * float(np.mean([ap for ap, _ in aps.values()])),
            "AP50": 100 * aps[0.5][0], "AP75": 100 * aps[0.75][0],
            "recall50": 100 * aps[0.5][1]}


def evaluate_detector(model, loader, decode, device):
    """``detection_metrics`` of the model on every image in ``loader``."""
    return detection_metrics(*predict_detections(model, loader, decode, device))


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

SIZE_BINS = (0.0, 0.25, 0.5, 0.75, 1.01)       # box height as a fraction of the image height


def best_f1_threshold(preds, gts, iou_threshold=0.5):
    """The score threshold that gives the best F1 score (balance of precision and recall)."""
    n_gt = sum(len(g) for g in gts)
    matched = [_match(p, g, iou_threshold) for p, g in zip(preds, gts)]
    scores = np.concatenate([s for s, _ in matched])
    tps = np.concatenate([t for _, t in matched])
    if len(scores) == 0 or n_gt == 0:
        return 0.5
    order = np.argsort(-scores, kind="stable")
    tp_cum = np.cumsum(tps[order])
    f1 = 2 * tp_cum / (np.arange(1, len(order) + 1) + n_gt)
    return float(scores[order][int(np.argmax(f1))])


def classify_errors(preds, gts, score_threshold, iou_threshold=0.5):
    """Label every detection above ``score_threshold`` and every true box.

    Detections: "correct" (matched to a true box), "duplicate" (overlaps a true
    box already found by a higher-scoring detection), "localisation" (best
    IoU 0.1 to 0.5: the right object, a poor box) or "background" (no
    pedestrian there, or one that is not labelled).
    True boxes: found or missed.

    Returns a list (one per image) of dicts with ``boxes``, ``scores``,
    ``labels`` (detections) and ``gt_found`` (bool per true box).
    """
    results = []
    for pred, gt in zip(preds, gts):
        keep = pred["scores"] >= score_threshold
        boxes, scores = pred["boxes"][keep], pred["scores"][keep]
        order = torch.argsort(scores, descending=True)
        boxes, scores = boxes[order], scores[order]
        labels, found = [], np.zeros(len(gt), dtype=bool)
        ious = box_iou(boxes, gt).numpy() if len(gt) else np.zeros((len(boxes), 0))
        for k in range(len(boxes)):
            free = np.where(found, -1.0, ious[k]) if len(gt) else np.array([])
            if len(gt) and free.max() >= iou_threshold:
                found[int(free.argmax())] = True
                labels.append("correct")
            elif len(gt) and ious[k].max() >= iou_threshold:
                labels.append("duplicate")
            elif len(gt) and ious[k].max() >= 0.1:
                labels.append("localisation")
            else:
                labels.append("background")
        results.append({"boxes": boxes, "scores": scores, "labels": labels, "gt_found": found})
    return results


def _crowded(gt, iou=0.1):
    """True for each box that overlaps another true box by more than ``iou``."""
    if len(gt) < 2:
        return np.zeros(len(gt), dtype=bool)
    ious = box_iou(gt, gt).numpy()
    np.fill_diagonal(ious, 0)
    return ious.max(axis=1) > iou


def error_analysis(preds, gts, image_size, score_threshold=None, plot=True):
    """Count the kinds of mistakes a detector makes, and which pedestrians it misses.

    ``score_threshold`` defaults to the one with the best F1 score. Prints
    the counts of each kind of detection, and the recall (fraction of true
    pedestrians found) by pedestrian height and for pedestrians that overlap
    another one ("crowded"). Returns the counts as a dict.
    """
    if score_threshold is None:
        score_threshold = best_f1_threshold(preds, gts)
    results = classify_errors(preds, gts, score_threshold)
    kinds = ("correct", "duplicate", "localisation", "background")
    counts = {k: sum(r["labels"].count(k) for r in results) for k in kinds}
    found = np.concatenate([r["gt_found"] for r in results])
    heights = np.concatenate([(g[:, 3] - g[:, 1]).numpy() / image_size for g in gts])
    crowded = np.concatenate([_crowded(g) for g in gts])
    counts["missed"] = int((~found).sum())

    groups = {}
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        sel = (heights >= lo) & (heights < hi)
        groups[f"height {lo:.2f}-{min(hi, 1):.2f}"] = (int(found[sel].sum()), int(sel.sum()))
    groups["not crowded"] = (int(found[~crowded].sum()), int((~crowded).sum()))
    groups["crowded"] = (int(found[crowded].sum()), int(crowded.sum()))

    print(f"Score threshold {score_threshold:.3f} (detections below it are ignored)")
    print(f"Detections : " + ", ".join(f"{k} {counts[k]}" for k in kinds))
    print(f"Pedestrians: {int(found.sum())} found, {counts['missed']} missed "
          f"(recall {100 * found.mean():.1f}%)")
    for name, (hit, total) in groups.items():
        rate = f"{100 * hit / total:5.1f}%" if total else "    -"
        print(f"  recall {name:<16}: {rate}  ({hit} of {total})")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
        colours = ["tab:blue", "tab:orange", "tab:purple", "tab:red", "tab:gray"]
        names = list(kinds) + ["missed"]
        bars = axes[0].bar(names, [counts[k] for k in names], color=colours)
        axes[0].bar_label(bars)
        axes[0].set(title="Detections and missed pedestrians", ylabel="count")
        labels = list(groups)
        rates = [100 * h / t if t else 0 for h, t in groups.values()]
        bars = axes[1].bar(labels, rates, color="tab:green")
        axes[1].bar_label(bars, labels=[f"{h}/{t}" for h, t in groups.values()], fontsize="small")
        axes[1].set(title="Recall by pedestrian height (fraction of image) and crowding",
                    ylabel="recall (%)", ylim=(0, 110))
        axes[1].tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        plt.show()
    return {"score_threshold": score_threshold, **counts,
            "recall by group": {k: v for k, v in groups.items()}}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_ERROR_COLOURS = {"correct": "tab:blue", "duplicate": "tab:orange",
                  "localisation": "tab:purple", "background": "tab:red"}


def _draw_box(ax, box, colour, style="-", width=2, text=None):
    x1, y1, x2, y2 = [float(v) for v in box]
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour,
                               linestyle=style, linewidth=width))
    if text:
        ax.text(x1 + 2, y1 + 2, text, color="white", fontsize=7, va="top",
                bbox=dict(facecolor=colour, edgecolor="none", pad=1, alpha=0.8))


def _grid(n, size=3.2):
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    return fig, list(axes.flat)


def show_boxes(images, boxes, n=8, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Show images (a batch tensor) with their true boxes in green."""
    n = min(n, len(images))
    fig, axes = _grid(n)
    for i in range(n):
        axes[i].imshow(denormalize(images[i], mean, std))
        for box in boxes[i]:
            _draw_box(axes[i], box, "tab:green")
        axes[i].set_title(f"{len(boxes[i])} pedestrians", fontsize="small")
    fig.tight_layout()
    plt.show()


def show_maps(images, maps, boxes=None, n=4, title="", grid=True, vmax=None,
              mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Overlay one map per image (e.g. an encoded target or a predicted score map).

    ``maps`` is a tensor [B, h, w] on the output grid of the detector; each
    cell is drawn over the ``stride`` x ``stride`` pixels it covers. With
    ``grid=True`` the cell borders are drawn too (only for coarse grids).
    True boxes (green) are drawn when ``boxes`` is given.
    """
    n = min(n, len(images))
    maps = maps.detach().float().cpu()
    fig, axes = _grid(n, size=3.6)
    for i in range(n):
        image = denormalize(images[i], mean, std)
        size = image.shape[0]
        h, w = maps[i].shape
        axes[i].imshow(image)
        axes[i].imshow(maps[i].numpy(), cmap="magma", alpha=0.55, vmin=0,
                       vmax=vmax if vmax is not None else max(float(maps[i].max()), 1e-6),
                       extent=(0, size, size, 0), interpolation="nearest")
        if grid and w <= 32:
            for k in range(1, w):
                axes[i].axvline(k * size / w, color="white", linewidth=0.3, alpha=0.5)
            for k in range(1, h):
                axes[i].axhline(k * size / h, color="white", linewidth=0.3, alpha=0.5)
        if boxes is not None:
            for box in boxes[i]:
                _draw_box(axes[i], box, "tab:green", width=1.5)
        axes[i].set_title(f"{title} ({h}x{w} grid)", fontsize="small")
    fig.tight_layout()
    plt.show()


def show_detections(dataset, preds, gts, indices=None, n=8, score_threshold=None, title=""):
    """Show images with their true boxes and the detections, coloured by kind.

    True boxes: green dashed (thick when missed). Detections above
    ``score_threshold`` (default: best F1): blue correct, orange duplicate,
    purple poor localisation, red background, with their score.
    ``preds`` and ``gts`` come from ``predict_detections`` with a loader over
    ``dataset`` that does not shuffle (validation or test), so they are in
    the same order as the dataset.
    """
    if score_threshold is None:
        score_threshold = best_f1_threshold(preds, gts)
    results = classify_errors(preds, gts, score_threshold)
    indices = list(range(min(n, len(dataset)))) if indices is None else list(indices)[:n]
    fig, axes = _grid(len(indices), size=3.6)
    for ax, i in zip(axes, indices):
        image, _ = dataset[i]
        ax.imshow(denormalize(image))
        for box, hit in zip(gts[i], results[i]["gt_found"]):
            _draw_box(ax, box, "tab:green", style="--", width=1.2 if hit else 3)
        for box, score, kind in zip(results[i]["boxes"], results[i]["scores"], results[i]["labels"]):
            _draw_box(ax, box, _ERROR_COLOURS[kind], text=f"{float(score):.2f}")
        missed = int((~results[i]["gt_found"]).sum())
        wrong = sum(k != "correct" for k in results[i]["labels"])
        ax.set_title(f"image {i}: {missed} missed, {wrong} wrong", fontsize="small")
    fig.suptitle(f"{title} (score threshold {score_threshold:.2f}); green dashed = true box, "
                 "thick = missed; blue correct, orange duplicate, purple localisation, red background",
                 fontsize="small")
    fig.tight_layout()
    plt.show()


def worst_images(preds, gts, n=8, score_threshold=None):
    """Indices of the ``n`` images with the most mistakes (missed + wrong detections)."""
    if score_threshold is None:
        score_threshold = best_f1_threshold(preds, gts)
    results = classify_errors(preds, gts, score_threshold)
    errors = [int((~r["gt_found"]).sum()) + sum(k != "correct" for k in r["labels"])
              for r in results]
    return [int(i) for i in np.argsort(-np.array(errors), kind="stable")[:n]]


def plot_precision_recall(preds, gts, iou_thresholds=(0.5, 0.75)):
    """Precision-recall curves at the given IoU thresholds (AP is the area under each)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for t in iou_thresholds:
        matched = [_match(p, g, t) for p, g in zip(preds, gts)]
        scores = np.concatenate([s for s, _ in matched])
        tps = np.concatenate([m for _, m in matched])
        order = np.argsort(-scores, kind="stable")
        tp_cum = np.cumsum(tps[order])
        recall = tp_cum / max(1, sum(len(g) for g in gts))
        precision = tp_cum / np.arange(1, len(order) + 1)
        ap, _ = average_precision(preds, gts, t)
        ax.plot(recall, precision, label=f"IoU {t:.2f}: AP {100 * ap:.1f}%")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.02),
           title="Precision-recall")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training history and final result
# ---------------------------------------------------------------------------

class DetectionHistory:
    """Record the loss terms, validation AP and learning rate of each epoch.

    ``log`` takes the training losses as a dict (one entry per loss term,
    e.g. ``{"score": 0.12, "box": 0.4}``) and the validation metrics from
    ``evaluate_detector``. The best epoch is the one with the highest
    validation AP.
    """

    def __init__(self, name="run", config=None, tensorboard=True, logdir="runs"):
        self.name, self.config = name, dict(config or {})
        self.epoch, self.lr, self.train_loss, self.val = [], [], [], []
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"{logdir}/{name}_{time.strftime('%Y%m%d-%H%M%S')}")

    def log(self, epoch, train_losses, val_metrics, lr=None, verbose=True):
        self.epoch.append(epoch)
        self.train_loss.append({k: float(v) for k, v in train_losses.items()})
        self.val.append(dict(val_metrics))
        self.lr.append(lr)
        if verbose:
            losses = "  ".join(f"{k} {v:.4f}" for k, v in self.train_loss[-1].items())
            lr_text = f"  lr {lr:.2e}" if lr is not None else ""
            best = "  *best" if self.is_best() else ""
            print(f"Epoch {epoch + 1:3d} | loss {losses} | val AP {val_metrics['AP']:5.1f}%  "
                  f"AP50 {val_metrics['AP50']:5.1f}%{lr_text}{best}")
        if self.writer:
            for k, v in self.train_loss[-1].items():
                self.writer.add_scalar(f"Loss/{k}", v, epoch)
            for k in ("AP", "AP50", "AP75"):
                self.writer.add_scalar(f"Validation/{k}", val_metrics[k], epoch)
            if lr is not None:
                self.writer.add_scalar("Learning rate", lr, epoch)
            self.writer.flush()

    def best_index(self):
        """Index of the epoch with the highest validation AP (the first one wins a tie)."""
        return int(np.argmax([v["AP"] for v in self.val]))

    def is_best(self):
        """True if the epoch logged last is the best so far (use it to keep the best weights)."""
        return bool(self.epoch) and self.best_index() == len(self.epoch) - 1

    def plot(self):
        """Plot the loss terms, validation AP and learning rate (saved in the notebook)."""
        if not self.epoch:
            print("Nothing to plot: no epochs have been logged yet.")
            return
        epochs = [e + 1 for e in self.epoch]
        has_lr = any(lr is not None for lr in self.lr)
        fig, axes = plt.subplots(1, 3 if has_lr else 2, figsize=(15 if has_lr else 11, 4))
        for key in self.train_loss[0]:
            axes[0].plot(epochs, [t[key] for t in self.train_loss], label=key)
        axes[0].set(title="Training loss terms", xlabel="epoch", ylabel="loss", yscale="log")
        for key in ("AP", "AP50", "AP75"):
            axes[1].plot(epochs, [v[key] for v in self.val], label=key)
        axes[1].set(title="Validation AP", xlabel="epoch", ylabel="AP (%)")
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
                  **{f"best val {k}": round(self.val[best][k], 1) for k in ("AP", "AP50", "AP75")}}
        width = max(len(k) for k in result)
        for key, value in result.items():
            print(f"{key:<{width}} : {value}")
        return result


@torch.no_grad()
def final_detection_result(model, history, test_loader, decode, device):
    """Evaluate the detector once on the test set and print the result block.

    Returns a dict with all values shown in the block.
    """
    from .models import compute_cost, format_count

    preds, gts = predict_detections(model, test_loader, decode, device)
    metrics = detection_metrics(preds, gts)
    image_size = test_loader.dataset.size
    cost = compute_cost(model, image_size, device, verbose=False)
    best = history.best_index() if history.epoch else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
    result = {
        "run": history.name,
        "image size": image_size,
        "test images": len(test_loader.dataset),
        "test pedestrians": sum(len(g) for g in gts),
        "epochs trained": len(history.epoch),
        "weights from epoch": history.epoch[best] + 1 if history.epoch else None,
        "val AP (%)": round(history.val[best]["AP"], 1) if history.epoch else None,
        "TEST AP (%)": round(metrics["AP"], 1),
        "test AP50 (%)": round(metrics["AP50"], 1),
        "test AP75 (%)": round(metrics["AP75"], 1),
        "test recall at IoU 0.5 (%)": round(metrics["recall50"], 1),
        "parameters": format_count(cost["params"]),
        "inference GFLOP per image": round(cost["inference_flops"] / 1e9, 2),
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
