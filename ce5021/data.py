"""Datasets, data loaders and image normalisation.

Typical use in a notebook::

    from ce5021.data import cifar10_loaders, CIFAR10_CLASSES

    train_loader, test_loader = cifar10_loaders(train_transforms, test_transforms,
                                                batch_size=128)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets

# Mean and standard deviation of the ImageNet training images (per RGB channel).
# Images are normalised with these values in the transforms, and any model
# pre-trained on ImageNet expects its input normalised the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def denormalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Undo normalisation so an image tensor can be displayed with plt.imshow.

    Args:
        image: normalised image tensor, shape [C, H, W] or [1, C, H, W].
        mean, std: the values used by v2.Normalize in the transforms.

    Returns:
        A numpy array of shape [H, W, C] with values clipped to 0..1.
        The input tensor is not modified.
    """
    image = image.detach().cpu()
    if image.dim() == 4 and image.shape[0] == 1:
        image = image[0]                              # drop the batch dimension
    if image.dim() != 3:
        raise ValueError(f"expected an image of shape [C, H, W], got {tuple(image.shape)}")

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    image = image * std + mean                        # reverse (x - mean) / std
    image = image.permute(1, 2, 0)                    # [C, H, W] -> [H, W, C]
    return np.clip(image.numpy(), 0, 1)


def cifar10_loaders(train_transform, test_transform, batch_size=128, root="data",
                    num_workers=2):
    """Download CIFAR-10 (first time only) and return (train_loader, test_loader).

    The training loader shuffles the images every epoch; the test loader
    keeps a fixed order.
    """
    train_data = datasets.CIFAR10(root=root, train=True, download=True,
                                  transform=train_transform)
    test_data = datasets.CIFAR10(root=root, train=False, download=True,
                                 transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


def _label_to_int(label):
    # MedMNIST labels are arrays of shape (1,); a plain int works directly
    # with nn.CrossEntropyLoss (no torch.squeeze needed in the training loop).
    return int(label[0])


def stratified_subset(labels, n, seed=0):
    """Indices of ``n`` items chosen at random with the same class proportions as ``labels``.

    Every class keeps at least one item. The same ``seed`` always gives the same subset.
    """
    labels = np.asarray(labels).reshape(-1)
    if n is None or n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    if n < len(classes):
        raise ValueError(f"n={n} is less than the number of classes ({len(classes)})")
    take = np.maximum(1, np.floor(counts * n / len(labels)).astype(int))
    # correct the rounding on the largest classes so exactly n are taken
    order = np.argsort(-counts)
    while take.sum() != n:
        step = 1 if take.sum() < n else -1
        for i in order:
            if take.sum() == n:
                break
            if take[i] + step >= 1:
                take[i] += step
    chosen = [rng.choice(np.flatnonzero(labels == c), k, replace=False)
              for c, k in zip(classes, take)]
    return np.sort(np.concatenate(chosen))


def class_weights(class_counts):
    """Loss weights that give every class the same total weight: n_images / (n_classes * count).

    Use with ``nn.CrossEntropyLoss(weight=class_weights(info["train_class_counts"]).to(device))``
    so mistakes on rare classes cost more than mistakes on common ones.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    return counts.sum() / (len(counts) * counts)


def medmnist_loaders(data_flag, train_transform, test_transform, batch_size=64,
                     size=28, n_train=None, balanced_sampling=False, seed=0,
                     root="data", num_workers=2):
    """Download a MedMNIST dataset and return (train_loader, val_loader, test_loader, info).

    Args:
        data_flag: dataset name in lower case, e.g. "dermamnist".
        size: image size, 28 (fast) or 64, 128, 224 (slower, more detail).
        n_train: number of training images to use (None: all of them). A random
            subset with the same class proportions, fixed by ``seed``.
        balanced_sampling: draw training images so every class is seen equally
            often (rare images are repeated). An epoch is still ``n_train`` images.

    ``info`` is a dict with keys ``task``, ``n_channels``, ``n_classes``,
    ``class_names``, ``n_train`` and ``train_class_counts``. Only multi-class
    and binary datasets are supported. The validation and test sets are
    always complete.
    """
    import medmnist   # installed separately: %pip install medmnist

    meta = medmnist.INFO[data_flag]
    if meta["task"] not in ("multi-class", "binary-class"):
        raise ValueError(f"{data_flag} is a '{meta['task']}' task; "
                         "choose a multi-class or binary dataset")
    DataClass = getattr(medmnist, meta["python_class"])
    n_classes = len(meta["label"])

    def split(name, transform):
        return DataClass(split=name, transform=transform, target_transform=_label_to_int,
                         download=True, size=size, root=root, mmap_mode="r")

    os.makedirs(root, exist_ok=True)
    train_data = split("train", train_transform)
    indices = stratified_subset(train_data.labels, n_train, seed)
    train_labels = np.asarray(train_data.labels).reshape(-1)[indices]
    counts = np.bincount(train_labels, minlength=n_classes)
    train_data = Subset(train_data, indices)

    if balanced_sampling:
        weights = 1.0 / counts[train_labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(indices), replacement=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
    val_loader = DataLoader(split("val", test_transform), batch_size=batch_size,
                            num_workers=num_workers)
    test_loader = DataLoader(split("test", test_transform), batch_size=batch_size,
                             num_workers=num_workers)

    info = {
        "task": meta["task"],
        "n_channels": meta["n_channels"],
        "n_classes": n_classes,
        "class_names": [meta["label"][str(i)] for i in range(n_classes)],
        "n_train": len(indices),
        "train_class_counts": counts.tolist(),
    }
    return train_loader, val_loader, test_loader, info
