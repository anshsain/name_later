# 🦺 SafetyWatch — Construction PPE Detection

Real-time helmet and safety compliance detection using YOLOv8 with two-phase transfer learning. Built end-to-end: data engineering, baseline training, fine-tuning, evaluation, and deployment.

**Live demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(update with your actual link)*

---

## Overview

SafetyWatch detects personal protective equipment (PPE) compliance on construction sites by identifying whether workers are wearing helmets. It classifies every detected head as either a helmet (compliant) or a bare head (violation), and reports an overall site compliance percentage.

| | |
|---|---|
| **Task** | Object detection (3 classes) |
| **Architecture** | YOLOv8n with two-phase transfer learning |
| **Dataset** | Hard Hat Workers (Roboflow), 7,000+ images |
| **mAP@50** | 64.3% |
| **Helmet AP@50** | 94.5% |
| **Head (violation) AP@50** | 96.4% |
| **Deployment** | Streamlit Community Cloud |

---

## Classes

| Class | Meaning |
|---|---|
| `helmet` | Worker wearing a hard hat — PPE compliant |
| `head` | Bare head detected — safety violation |
| `person` | Full body bounding box |

---

## Architecture & Training Strategy

The model uses a **two-phase transfer learning** approach rather than training end-to-end from scratch:

**Phase 1 — Warm-up (15 epochs).** The first 9 backbone layers of YOLOv8n are frozen, so only the neck and detection head are trained. This stabilizes the new task-specific layers without disturbing the pretrained COCO features in the backbone.

**Phase 2 — Fine-tuning (50 epochs).** All layers are unfrozen and trained end-to-end at a 10x lower learning rate (1e-4 vs 1e-3), allowing the backbone to adapt to the construction-site domain without catastrophic forgetting.

This freeze-then-unfreeze strategy is a standard transfer learning technique that gives more stable convergence than fine-tuning every layer from the first epoch.

---

## Results

### Per-class performance

| Class | AP@50 | Notes |
|---|---|---|
| helmet | 94.5% | Strong, well-represented class |
| head | 96.4% | Strong, well-represented class |
| person | 2.0% | Severely limited by class imbalance (see below) |

### Overall metrics

| Metric | Value |
|---|---|
| mAP@50 | 64.3% |
| mAP@50-95 | 42.6% |
| Precision | 62.1% |
| Recall | 61.6% |
| Inference speed | ~4ms/image (GPU) |

### Confusion matrix

Helmet and head are rarely confused with each other — only about 2% of helmet instances are misclassified as head and vice versa, which is the property that matters most for a compliance-monitoring use case.

---

## Known limitation: class imbalance

The `person` class has only 615 training instances versus 16,833 for `head` — a **27.4x imbalance**. This is a property of the source dataset, not a modeling failure: the dataset was originally curated for helmet/head detection, with `person` boxes annotated inconsistently as a secondary class.

This was confirmed through evaluation, not assumed:
- Per-class AP and precision-recall curves showed `person` recall near zero across all confidence thresholds
- Instance counts per split confirmed the imbalance ratio
- GRAD-CAM activations remained correctly localized on heads, ruling out a backbone or label-quality issue

For production use, the `person` class should either be dropped, with `helmet`/`head` used as the two real detection targets, or rebalanced with additional annotated data.

---

## Project structure

```
.
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── best.pt                   # Trained model weights (YOLOv8n, two-phase)
└── notebooks/
    ├── layer1_setup.ipynb       # Dataset download, EDA, augmentation preview
    ├── layer2_baseline.ipynb    # Baseline YOLOv8n training (50 epochs)
    ├── layer3_resnet50.ipynb    # Two-phase transfer learning
    ├── layer4_gradcam.ipynb     # GRAD-CAM, PR curves, confusion matrix, failure analysis
    └── layer5_deploy.ipynb      # Hugging Face / Streamlit deployment automation
```

---

## Running locally

```bash
git clone https://github.com/your-username/safetywatch-ppe.git
cd safetywatch-ppe
pip install -r requirements.txt
streamlit run app.py
```

The app loads `best.pt` from the repo root automatically — no path configuration needed.

---

## App features

**Image detection.** Upload a JPG/PNG, run inference, and view bounding boxes with class labels and confidence scores. The sidebar shows a live compliance summary: helmet count, violation count, and overall compliance percentage.

**Video detection.** Upload an MP4/AVI/MOV file for frame-by-frame detection, with a downloadable annotated output video.

**Model insights.** Static dashboard of per-class AP, overall metrics, and a summary of the two-phase training strategy.

---

## Reproducing the training pipeline

The four training notebooks are designed to run sequentially in Google Colab with a T4 GPU, using Google Drive for persistent storage between sessions.

1. **Layer 1** downloads the dataset via the Roboflow API and runs exploratory data analysis (class distribution, bounding box size statistics, augmentation preview).
2. **Layer 2** trains the baseline YOLOv8n model for 50 epochs and saves benchmark metrics to `baseline_benchmark.json`.
3. **Layer 3** loads that benchmark and runs the two-phase fine-tuning strategy described above, then compares results against the baseline.
4. **Layer 4** produces GRAD-CAM heatmaps, precision-recall curves, a confusion matrix, and a class imbalance analysis — going beyond headline metrics to explain *why* the model performs the way it does.

Each notebook re-derives all file paths at runtime, so they can be run independently as long as the previous layer's outputs exist in Drive.

---

## Tech stack

PyTorch · Ultralytics YOLOv8 · OpenCV · Streamlit · Roboflow · Google Colab (T4 GPU)
