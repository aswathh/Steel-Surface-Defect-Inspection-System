# Steel Surface Defect Inspection System

An end-to-end computer vision project on steel surface defects, built on the NEU-DET dataset. It covers image classification, object detection and anomaly detection, in both PyTorch and TensorFlow, with model export and a Streamlit demo.

## Overview

| Task | Question it answers | Models |
|---|---|---|
| Classification | Which defect is this? | SimpleCNN, ResNet18 (PyTorch); SimpleCNN, MobileNetV2 (Keras) |
| Object detection | Where is the defect, and which type? | YOLO11n, Faster R-CNN |
| Anomaly detection | Does this look different from normal? | Convolutional autoencoder, PatchCore |
| Deployment | Can the models run outside the training code? | ONNX, TFLite, Streamlit app |

## Dataset

NEU-DET: 200x200 grayscale images of hot-rolled steel strips with six defect classes:
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.

- Training: 1440 images (240 per class)
- Validation: 360 images (60 per class)
- Annotations: Pascal VOC XML bounding boxes
- Note: one annotation file in the validation split has no matching image, so it is skipped during data preparation.

The dataset is not stored in this repository. Download it from Kaggle and place it at `data/NEU-DET`, keeping the `train` and `validation` folders, each with `annotations` and `images`.

## Project structure

```
Steel Surface Defect Inspection System/
    app/                    Streamlit demo
    data/                   dataset (not tracked by Git)
    models/                 saved weights and exports (not tracked by Git)
    notebooks/              EDA and TensorFlow/Keras experiments
    outputs/                YOLO training runs (not tracked by Git)
    src/
        dataset.py          PyTorch classification dataset
        transforms.py       Albumentations augmentation
        model.py            SimpleCNN and ResNet18 builders
        train.py            classification training loop
        prepare_yolo.py     XML to YOLO label conversion
        train_yolo.py       YOLO training
        export_yolo.py      YOLO export to ONNX
        frcnn_dataset.py    Faster R-CNN dataset
        frcnn_model.py      Faster R-CNN builder
        train_frcnn.py      Faster R-CNN training
        eval_frcnn.py       Faster R-CNN evaluation (mAP)
        anomaly_data.py     data loading for anomaly detection
        anomaly_model.py    convolutional autoencoder
        train_anomaly.py    autoencoder training and AUROC
        patchcore.py        PatchCore implementation and AUROC
```

## Setup

The project uses `uv` with Python 3.12.

```bash
uv sync
```

Main libraries: PyTorch, torchvision, TensorFlow, Ultralytics, ONNX Runtime, Albumentations, OpenCV, scikit-learn, torchmetrics, Streamlit.

## Part 1: Classification

### PyTorch

- Images resized to 224x224 and normalized with ImageNet statistics.
- Training augmentation with Albumentations: horizontal and vertical flips, brightness and contrast, small rotations. Validation uses resize and normalize only.
- SimpleCNN: three convolution blocks with batch normalization, then global average pooling and a linear layer.
- ResNet18: pretrained on ImageNet, backbone frozen, only the final layer trained.
- Adam optimizer, learning rate 1e-3, 15 epochs.

### TensorFlow / Keras

- SimpleCNN with the same structure as the PyTorch version.
- MobileNetV2 with a frozen pretrained base and a new classification head, 10 epochs.
- Exported to TFLite with default optimization.

### Classification results

| Model | Framework | Format | Size | Speed (CPU) | Val accuracy |
|---|---|---|---|---|---|
| SimpleCNN | PyTorch | ONNX | 0.39 MB | 3.7 ms | 74.7% |
| ResNet18 (transfer) | PyTorch | ONNX | 44.79 MB | 12.6 ms | 96.9% |
| SimpleCNN | Keras | TFLite | 0.11 MB | 5.7 ms | 52.5% |
| MobileNetV2 (transfer) | Keras | TFLite | 2.54 MB | 54.6 ms | 99.4% |

Observations:

- Transfer learning clearly beats training from scratch in both frameworks.
- The PyTorch SimpleCNN confused crazing with patches (almost all crazing images were misclassified). ResNet18 was near perfect, with its main error being inclusion predicted as scratches.
- MobileNetV2 is the most accurate and about 17 times smaller than ResNet18.
- ONNX and TFLite timings are not directly comparable, since they are optimized for different runtimes.
- ONNX outputs were checked against PyTorch outputs and matched.

## Part 2: Object detection

### Data preparation

`prepare_yolo.py` converts the Pascal VOC boxes (pixel corners) into YOLO format: one text file per image with `class x_center y_center width height`, all as fractions of the image size. It also writes `data.yaml`.

### Models

- YOLO11n (Ultralytics): pretrained weights, 30 epochs, image size 320, batch size 16, trained on CPU. Exported to ONNX.
- Faster R-CNN (torchvision, ResNet50-FPN): pretrained on COCO, classification head replaced for 6 classes plus background, 5 epochs, image size 320, SGD with momentum, trained on CPU.

### Detection results

| | YOLO11n | Faster R-CNN |
|---|---|---|
| Type | single-stage | two-stage |
| mAP50 | 0.734 | 0.731 |
| mAP50-95 | 0.395 | 0.351 |
| Speed (CPU) | about 11 ms per image | about 516 ms per image |
| Model size | 10.1 MB (ONNX) | about 158 MB (PyTorch weights) |
| Training | 30 epochs | 5 epochs |

YOLO11n per-class mAP50:

| Class | mAP50 |
|---|---|
| patches | 0.929 |
| scratches | 0.861 |
| pitted_surface | 0.808 |
| inclusion | 0.803 |
| crazing | 0.507 |
| rolled-in_scale | 0.496 |

Faster R-CNN per-class mAP (averaged over IoU 0.5 to 0.95): crazing 0.142, inclusion 0.395, patches 0.554, pitted_surface 0.417, rolled-in_scale 0.186, scratches 0.409.

Observations:

- Both detectors reach almost the same mAP50. YOLO has tighter boxes (higher mAP50-95) and is roughly 47 times faster on CPU, which makes it the better fit for real-time inspection.
- Both models find crazing and rolled-in_scale hardest. These defects are faint and spread out, with low-contrast boundaries.
- The comparison is not perfectly fair. Faster R-CNN trained for only 5 epochs and its loss was still falling, and the two speeds were measured with different procedures, so the speed gap is approximate.

## Part 3: Anomaly detection

Anomaly detection trains on normal images only and flags anything that looks different. NEU-DET contains no defect-free images, so as a stand-in experiment the `patches` class was treated as "normal" and the other five classes as "anomalous". AUROC measures how well the scores separate the two groups (1.0 is perfect, 0.5 is random guessing).

| Method | Idea | Normal class | AUROC |
|---|---|---|---|
| Convolutional autoencoder | rebuild the image, use reconstruction error as the score | patches | 0.252 |
| PatchCore (ResNet18 features) | compare patch features against a memory bank of normal patches | patches | 0.457 |

Both results are below 0.5, so neither separates the groups usefully in this setup.

Likely reasons (not yet confirmed on clean data):

- NEU-DET has no defect-free class. Patches is itself a defect, so "patches versus the rest" is not a true normal-versus-anomaly test.
- The autoencoder's error follows image texture and contrast, not abnormality. Busy patches images were rebuilt worse than smoother defect images.
- The PatchCore memory bank was a random sample of 20000 patch features, so many normal patches were left out.

A proper evaluation needs a dataset with defect-free training images, such as MVTec AD. Running the same code on one MVTec AD category is the natural next step. The anomaly models are therefore not included in the demo app.

## Demo app

The Streamlit app lets you upload a steel surface image and returns:

1. The predicted defect class and confidence from ResNet18.
2. YOLO11n detections drawn on the image.

```bash
uv run streamlit run app/app.py
```

The app expects `models/resnet18.pt` and `outputs/yolo_run/weights/best.pt` to exist, so run the training scripts first. To test it, upload any image from `data/NEU-DET/validation/images`.

## How to reproduce

```bash
# Classification (PyTorch)
uv run python src/train.py

# Detection: YOLO
uv run python src/prepare_yolo.py
uv run python src/train_yolo.py
uv run python src/export_yolo.py

# Detection: Faster R-CNN
uv run python src/train_frcnn.py
uv run python src/eval_frcnn.py

# Anomaly detection
uv run python src/train_anomaly.py
uv run python src/patchcore.py

# Demo
uv run streamlit run app/app.py
```

Model weights are excluded from Git (`models/*.pt`, `outputs/`) because some files exceed GitHub's 100 MB limit, so they must be regenerated with the commands above.

## Key takeaways

- Transfer learning is the biggest single improvement for small datasets, in both PyTorch and Keras.
- For detection, a single-stage model (YOLO) matched a two-stage model (Faster R-CNN) on mAP50 while being far faster and smaller.
- Plain reconstruction-based anomaly detection fails when the notion of "normal" is not clean. Evaluation data matters as much as the method.

## Limitations and next steps

- Evaluate PatchCore and the autoencoder on MVTec AD for a proper anomaly detection benchmark.
- Train Faster R-CNN for more epochs for a fairer comparison with YOLO.
- Improve crazing and rolled-in_scale detection with higher input resolution, longer training or a larger YOLO variant.
- Add SSD or EfficientDet to extend the detector comparison.
- Containerize the app and serve the ONNX models through FastAPI.