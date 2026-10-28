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
| `ce5021.data` | `cifar10_loaders`, `medmnist_loaders` (image size, number of training images, balanced sampling), `stratified_subset`, `class_weights`, `denormalize`, `IMAGENET_MEAN`, `IMAGENET_STD`, `CIFAR10_CLASSES` |
| `ce5021.models` | `build_model` (pre-trained torchvision classifier with a new head; strategies `head`, `partial`, `full`, `scratch`; `train_from` chooses where `partial` starts), `show_layers` (blocks of a model in order, trained or frozen), `compute_cost` (parameters and FLOPs), `ARCHITECTURES` |
| `ce5021.train` | `get_device`, `evaluate`, `evaluate_balanced`, `predict`, `History` (`log`, `is_best`, `plot`, `summary`), `final_evaluation` |
| `ce5021.plots` | `show_batch`, `show_images`, `show_predictions`, `plot_class_counts`, `plot_confusion_matrix`, `plot_tensorboard_logs` |
| `ce5021.detection` | `pennfudan_loaders` (letterboxed Penn-Fudan, boxes per image), `build_detector` (single-stage detector from design options: backbone, stride, neck, targets, score loss, decoding; each option is a function such as `encode_gaussian`, `focal_score_loss`, `keep_peaks`), `build_backbone`, `predict_detections`, `evaluate_detector`, `detection_metrics` (COCO AP, AP50, AP75), `error_analysis`, `worst_images`, `show_boxes`, `show_maps`, `show_detections`, `plot_precision_recall`, `DetectionHistory`, `final_detection_result` |
| `ce5021.segmentation` | `atr_loaders` (ATR human parsing from Hugging Face, resized once and cached, `n_train` subset), `RandomFlipLeftRight` (flip that swaps left/right classes), `resnet_stages` (ImageNet ResNet stages for an encoder with skip connections), `evaluate_segmentation`, `predict_segmentation` (mIoU, pixel accuracy, class IoU), `show_segmentation`, `plot_class_pixels`, `plot_class_iou`, `plot_confusion`, `worst_images`, `SegmentationHistory`, `final_segmentation_result` |

In a notebook, `help(evaluate)` shows the documentation of a function and
`evaluate??` shows its source code.
