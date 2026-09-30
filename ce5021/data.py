"""Datasets, data loaders and image normalisation.

Typical use in a notebook::

    from ce5021.data import cifar10_loaders, CIFAR10_CLASSES

    train_loader, test_loader = cifar10_loaders(train_transforms, test_transforms,
                                                batch_size=128)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
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


def medmnist_loaders(data_flag, train_transform, test_transform, batch_size=64,
                     size=28, root="data", num_workers=2):
    """Download a MedMNIST dataset and return (train_loader, val_loader, test_loader, info).

    Args:
        data_flag: dataset name in lower case, e.g. "bloodmnist", "pneumoniamnist".
        size: image size, 28 (fast) or 64, 128, 224 (slower, more detail).

    ``info`` is a dict with keys ``task``, ``n_channels``, ``n_classes`` and
    ``class_names``. Only multi-class and binary datasets are supported.
    """
    import medmnist   # installed separately: %pip install medmnist

    meta = medmnist.INFO[data_flag]
    if meta["task"] not in ("multi-class", "binary-class"):
        raise ValueError(f"{data_flag} is a '{meta['task']}' task; "
                         "choose a multi-class or binary dataset")
    DataClass = getattr(medmnist, meta["python_class"])

    def split(name, transform):
        return DataClass(split=name, transform=transform, target_transform=_label_to_int,
                         download=True, size=size, root=root, mmap_mode="r")

    os.makedirs(root, exist_ok=True)
    train_loader = DataLoader(split("train", train_transform), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(split("val", test_transform), batch_size=batch_size,
                            num_workers=num_workers)
    test_loader = DataLoader(split("test", test_transform), batch_size=batch_size,
                             num_workers=num_workers)

    info = {
        "task": meta["task"],
        "n_channels": meta["n_channels"],
        "n_classes": len(meta["label"]),
        "class_names": [meta["label"][str(i)] for i in range(len(meta["label"]))],
    }
    return train_loader, val_loader, test_loader, info
