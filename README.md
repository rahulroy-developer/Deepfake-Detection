# Deepfake Image Detection (Multi-Domain Forensic Pipeline)

A lightweight, explainable machine learning pipeline for detecting deepfake face images. Instead of relying on heavy black-box CNNs, this project isolates the facial region to eliminate background data leakage, extracts spatial and frequency-domain manipulation artifacts, and classifies images using a regularized XGBoost ensemble.

## Pipeline Overview

1. **Strict Face Isolation:** Uses MediaPipe Face Landmarker to detect facial landmarks, crops the face bounding box with a 10-pixel margin (preserving jawline and forehead blending boundaries), resizes it to `256x256`, and converts it to grayscale.
2. **Multi-Domain Feature Extraction (1,614 Features):**
   - **Uniform LBP (`P=8, R=1`) — 10 dims:** Captures over-smoothed skin and micro-texture anomalies.
   - **Canny & Sobel Edges — 4 dims:** Measures edge density and gradient magnitudes to detect blending seams and blur.
   - **2D FFT Radial Profile — 32 dims:** Exposes periodic high-frequency upsampling artifacts in the frequency domain.
   - **HOG Descriptor — 1,568 dims:** Captures directional shading and local structural gradients (`orientations=8`, `pixels_per_cell=(32, 32)`, `cells_per_block=(2, 2)`).
3. **Scaling & Classification:** Applies Z-score normalization (`StandardScaler`, fitted strictly on the training split) and classifies faces using a regularized `XGBClassifier` (`n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).

## Prototype Results

Evaluated on a class-balanced 5,000-image subset of the **140k Real and Fake Faces** dataset (3,000 train, 998 validation, 998 test after automatic filtering of images with no detectable face):

| Metric | Score |
| :--- | :---: |
| Validation Accuracy | 77.45% |
| Test Accuracy | 75.85% |
| Precision (Fake) | 75.44% |
| Recall / Sensitivity (Fake) | 76.80% |
| Specificity (Real) | 74.90% |
| F1-Score (Fake) | 76.12% |
| ROC-AUC | 0.841 |

## Project Structure

```text
.
├── deepfake_v2.ipynb                   # End-to-end preprocessing, training, evaluation & inference notebook
├── face_landmarker.task                # MediaPipe Face Landmarker model asset
├── real_fake_xgboost_clean.pkl         # Trained XGBoost model
├── feature_scaler_clean.pkl            # Fitted StandardScaler
├── artifact_only_features_subset.npz   # Extracted feature arrays (optional / gitignored)
├── assets/
│   ├── confusion_matrix.png            # Test set confusion matrix
│   └── roc_curve.png                   # Test set ROC curve
└── README.md
