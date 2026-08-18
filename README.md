# STRAIGHT 🩻

Landmark-regression method for detecting patient rotation in X-ray images, benchmarked against a classification CNN and a panel of human raters.

## Overview

STRAIGHT predicts three landmark x-coordinates (`X1`, `X2`, `X3`) on an X-ray image using an HRNet-W48 backbone. From these points it derives a rotation index:

```
distance_left  = X2 - X3
distance_right = X3 - X1
alpha = (distance_right - distance_left) / (distance_right + distance_left)
```

`alpha` is then thresholded (±0.2) to classify the image as **Left rotate**, **Normal**, or **Right rotate**. This repository also trains a direct 3-class classification CNN (same HRNet-W48 backbone) as a baseline, and compares both models against human rater performance (label-based and visual-based rating).

## Repository Structure

```
STRAIGHT/
├── train_STRAIGHT.ipynb            # Train/evaluate the landmark-regression (STRAIGHT) model
├── train_Classification_model.ipynb# Train/evaluate the 3-class classification CNN baseline
├── analyze.ipynb                   # Compare STRAIGHT vs classification model (metrics, ROC, Grad-CAM)
├── human_analyze.ipynb             # Compare models against human rater agreement/performance
└── requirements.txt
```

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.8+ and a CUDA-capable GPU is recommended for training.

## Training

**Landmark regression (STRAIGHT):**
Run [train_STRAIGHT.ipynb](train_STRAIGHT.ipynb) — trains an HRNet-W48 backbone with a regression head to predict `X1`, `X2`, `X3`, and reports MAE against ground-truth landmarks.

**Classification baseline:**
Run [train_Classification_model.ipynb](train_Classification_model.ipynb) — trains an HRNet-W48 classifier directly on the 3 rotation classes (Left rotate / Normal / Right rotate).

Both notebooks expect a CSV-indexed image dataset split into `Train` / `Validation` / `Test` folders — update the hard-coded dataset paths at the top of each notebook to match your local data layout.

## Evaluation

- [analyze.ipynb](analyze.ipynb) — loads both trained models, computes MAE/accuracy/sensitivity/specificity/precision/F1, confusion matrices, bootstrap ROC-AUC with 95% CI, and Grad-CAM visualizations; saves qualitative comparison figures per outcome category (both right, both wrong, etc.).
- [human_analyze.ipynb](human_analyze.ipynb) — computes inter-rater ICC and weighted Cohen's kappa for human raters, per-rater performance vs. ground truth, model-vs-human comparison tables, statistical tests (Cochran's Q, McNemar's), and radar/ROC plots. Expects rater result CSVs under a `Result/` directory (not included in this repo).

## Requirements

See [requirements.txt](requirements.txt). Core dependencies: PyTorch, torchvision, timm (HRNet-W48), segmentation-models-pytorch, OpenCV, scikit-learn, pingouin, pytorch-grad-cam.

## License

TBD
