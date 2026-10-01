"""Pre-trained image classifiers for transfer learning, and their compute cost.

Typical use in a notebook::

    from ce5021.models import build_model, compute_cost

    model = build_model("resnet18", n_classes=7, strategy="head").to(device)
    cost = compute_cost(model, image_size=64, device=device)

The models are the standard ImageNet classifiers from torchvision. The final
ImageNet layer (1000 classes) is replaced by a new *head* for our classes, and
the *strategy* decides which layers are trained:

    "head"     only the new head; the pre-trained layers are frozen
               (the network is used as a fixed feature extractor)
    "partial"  the last stage of the network and the head
    "full"     every layer (fine-tuning), starting from the ImageNet weights
    "scratch"  every layer, starting from random weights (no transfer learning)
"""

from torch import nn
from torchvision import models as tv_models

# For each architecture: the attribute path of the final ImageNet layer, the
# first parameter of the last stage (where "partial" starts training), and the
# smallest input image size the network accepts.
ARCHITECTURES = {
    "resnet18":           {"head": "fc",            "last_stage": "layer4",               "min_size": 28},
    "resnet50":           {"head": "fc",            "last_stage": "layer4",               "min_size": 28},
    "vgg16_bn":           {"head": "classifier.6",  "last_stage": "features.34",          "min_size": 32},
    "densenet121":        {"head": "classifier",    "last_stage": "features.denseblock4", "min_size": 32},
    "mobilenet_v3_large": {"head": "classifier.3",  "last_stage": "features.13",          "min_size": 28},
    "efficientnet_b0":    {"head": "classifier.1",  "last_stage": "features.7",           "min_size": 28},
    "convnext_tiny":      {"head": "classifier.2",  "last_stage": "features.6",           "min_size": 32},
}
STRATEGIES = ("head", "partial", "full", "scratch")
HEADS = ("linear", "mlp")


def _get(module, path):
    for name in path.split("."):
        module = module[int(name)] if name.isdigit() else getattr(module, name)
    return module


def _set(module, path, new):
    parent, _, name = path.rpartition(".")
    parent = _get(module, parent) if parent else module
    if name.isdigit():
        parent[int(name)] = new
    else:
        setattr(parent, name, new)


def make_head(in_features, n_classes, head="linear", hidden=256, dropout=0.5):
    """The new classifier: one linear layer, or "mlp" (linear -> ReLU -> dropout -> linear)."""
    if head == "linear":
        return nn.Linear(in_features, n_classes)
    if head == "mlp":
        return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(),
                             nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    raise ValueError(f"head must be one of {HEADS}, not {head!r}")


def build_model(arch, n_classes, strategy="head", head="linear", hidden=256, dropout=0.5):
    """Load a torchvision classifier, replace its last layer and choose what is trained.

    Args:
        arch: one of ``ARCHITECTURES`` (e.g. "resnet18", "efficientnet_b0").
        n_classes: number of classes in your dataset.
        strategy: "head", "partial", "full" or "scratch" (see the module help).
        head: "linear" or "mlp"; ``hidden`` and ``dropout`` set up the mlp head.

    The ImageNet weights are downloaded the first time (not for "scratch").
    Parameters that are not trained have ``requires_grad = False``.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {list(ARCHITECTURES)}, not {arch!r}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
    spec = ARCHITECTURES[arch]

    weights = None if strategy == "scratch" else "DEFAULT"
    model = getattr(tv_models, arch)(weights=weights)

    old_head = _get(model, spec["head"])
    _set(model, spec["head"], make_head(old_head.in_features, n_classes, head, hidden, dropout))

    # Choose the trained parameters. named_parameters() lists them in network
    # order, so "partial" trains everything from the start of the last stage on.
    trainable = strategy in ("full", "scratch")
    for name, param in model.named_parameters():
        if strategy == "partial" and name.startswith(spec["last_stage"] + "."):
            trainable = True
        param.requires_grad = trainable or name.startswith(spec["head"] + ".")

    model.ce5021_info = {"arch": arch, "strategy": strategy, "head": head,
                         "min_size": spec["min_size"]}
    return model


def format_count(n):
    """11180103 -> "11.18 M", 3591 -> "3,591"."""
    return f"{n / 1e6:.2f} M" if n >= 100_000 else f"{n:,}"


def count_parameters(model):
    """Return (total, trainable) number of parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _layer_flops(module, output):
    """Forward FLOPs for one image through a convolution or linear layer (2 per multiply-add)."""
    per_image = output.numel() // output.shape[0]
    if isinstance(module, nn.Conv2d):
        k_h, k_w = module.kernel_size
        return 2 * per_image * (module.in_channels // module.groups) * k_h * k_w
    return 2 * per_image * module.in_features            # nn.Linear


def compute_cost(model, image_size, device="cpu", verbose=True):
    """Measure the parameters and floating-point operations (FLOPs) of a model.

    Returns a dict with:
        params, trainable_params
        inference_flops   FLOPs to classify one image (forward pass)
        train_flops       FLOPs to train on one image: the forward pass, plus a
                          backward pass through the layers being trained, which
                          costs about twice their forward FLOPs (gradients for
                          their inputs and for their weights). Frozen layers
                          only run forward, so freezing makes training cheaper.
        image_size

    FLOPs are counted for the convolution and linear layers, which account for
    nearly all of the computation. Multiply ``train_flops`` by the number of
    training images and epochs to get the compute used for a training run.
    """
    import torch

    info = getattr(model, "ce5021_info", {})
    if image_size < info.get("min_size", 0):
        raise ValueError(f"{info['arch']} needs images of at least {info['min_size']}x"
                         f"{info['min_size']} pixels; image_size is {image_size}")

    totals = {"all": 0, "trained": 0}

    def hook(module, inputs, output):
        flops = _layer_flops(module, output)
        totals["all"] += flops
        if any(p.requires_grad for p in module.parameters()):
            totals["trained"] += flops

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, image_size, image_size, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    inference = totals["all"]
    train = totals["all"] + 2 * totals["trained"]

    total, trainable = count_parameters(model)
    cost = {"params": total, "trainable_params": trainable, "inference_flops": inference,
            "train_flops": train, "image_size": image_size, **info}
    if verbose:
        name = f"{info['arch']} (strategy {info['strategy']}, head {info['head']})" if info else "model"
        print(f"{name}, {image_size}x{image_size} images")
        print(f"  parameters : {format_count(total)} total, {format_count(trainable)} trained")
        print(f"  inference  : {inference / 1e9:.3f} GFLOP per image")
        print(f"  training   : {train / 1e9:.3f} GFLOP per training image (forward + backward)")
    return cost
