[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tafsir99-gg/breast-cancer-detection-pytorch-training-pipeline/blob/main/cancer_detection_%28pytorch_pipeline%29.ipynb)

# 🧬 Breast Cancer Detection — PyTorch Neural Network Pipeline

A fully documented deep learning pipeline for binary classification of breast tumors (malignant vs. benign) using PyTorch, built on the classic Wisconsin Breast Cancer dataset from scikit-learn.

---

## 📌 Overview

This project trains a feedforward neural network to classify breast tumor samples as **Malignant (0)** or **Benign (1)** based on 30 numerical features extracted from cell nucleus measurements. The pipeline covers everything from data loading and preprocessing to model training, evaluation, and inference.

---

## 📂 Project Structure

```
cancer_detection_pytorch_pipeline/
│
├── cancer_detection__pytorch_pipeline_.ipynb   # Main Colab notebook
└── README.md                                    # Project documentation
```

---

## 📊 Dataset

**Source:** `sklearn.datasets.load_breast_cancer` (built-in, no download needed)

| Property | Value |
|---|---|
| Total samples | 569 |
| Features | 30 numerical |
| Classes | 2 (Malignant / Benign) |
| Malignant (0) | 212 samples |
| Benign (1) | 357 samples |

The 30 features are computed from digitized images of fine needle aspirates (FNA) of breast masses. They describe characteristics of cell nuclei such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension — each measured as mean, standard error, and worst value.

---

## 🧠 Model Architecture

```
Input (30)
  └── Linear(30 → 64) → BatchNorm1d → ReLU → Dropout(0.3)
        └── Linear(64 → 32) → BatchNorm1d → ReLU → Dropout(0.3)
              └── Linear(32 → 16) → ReLU
                    └── Linear(16 → 1) → Sigmoid
```

**Output:** A single probability value between 0 and 1. Values ≥ 0.5 are classified as Benign, values < 0.5 as Malignant.

| Component | Purpose |
|---|---|
| `BatchNorm1d` | Stabilizes and speeds up training by normalizing layer inputs |
| `ReLU` | Activation function — introduces non-linearity |
| `Dropout(0.3)` | Randomly drops 30% of neurons during training to prevent overfitting |
| `Sigmoid` | Squashes the final output to a [0, 1] probability |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss Function | `BCELoss` (Binary Cross Entropy) |
| Optimizer | `Adam` (lr=1e-3, weight_decay=1e-4) |
| LR Scheduler | `ReduceLROnPlateau` (factor=0.5, patience=5) |
| Gradient Clipping | max_norm=1.0 |
| Epochs | 100 |
| Batch Size | 32 |
| Train/Test Split | 80% / 20% (stratified) |

---

## 🔄 Pipeline Steps

1. **Imports** — PyTorch, scikit-learn, NumPy, Matplotlib, Seaborn
2. **Data Loading** — Load breast cancer dataset, explore shape and class distribution
3. **Preprocessing** — Stratified split, `StandardScaler` (fit on train only), convert to tensors
4. **DataLoaders** — Wrap tensors in `TensorDataset`, create batched `DataLoader`s
5. **Model Definition** — Define `BreastCancerNet` with `nn.Module`
6. **Training Setup** — Loss function, Adam optimizer, LR scheduler
7. **Training Loop** — Full training with backprop, gradient clipping, and per-epoch metrics
8. **Training Curves** — Plot loss and accuracy over epochs
9. **Evaluation** — Accuracy, Precision, Recall, F1-score, AUC-ROC, ROC Curve
10. **Confusion Matrix** — Heatmap of true vs predicted labels
11. **Inference** — Run predictions on 5 random test samples

---

## 📈 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~97% |
| AUC-ROC | ~0.99 |

Training and validation loss track closely throughout training with no signs of overfitting, confirmed by the near-zero gap between training and validation accuracy curves.

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Click the notebook file in this repository
2. Click **"Open in Colab"** at the top
3. Run all cells in order (`Runtime → Run all`)

No setup needed — all dependencies are pre-installed in Colab.

### Option 2 — Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install torch scikit-learn numpy matplotlib seaborn pandas

# 3. Launch Jupyter
jupyter notebook cancer_detection__pytorch_pipeline_.ipynb
```

---

## 🛠 Dependencies

```
torch
scikit-learn
numpy
matplotlib
seaborn
pandas
```

All of these come pre-installed on Google Colab. For local use, install with:

```bash
pip install torch scikit-learn numpy matplotlib seaborn pandas
```

---

## ⚠️ Known Issue

In the preprocessing cell, `X_test` is scaled using `scaler.fit_transform()` instead of `scaler.transform()`. The correct approach is:

```python
# ✅ Correct
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   # use transform only, not fit_transform

# ❌ Current (avoidable data leakage on test set)
X_test = scaler.fit_transform(X_test)
```

Using `fit_transform` on the test set fits a new scaler to test data, which is a form of data leakage. It doesn't break the model but it means the test set is scaled independently rather than using the training distribution.

---

## 📚 Concepts Used

- **StandardScaler** — Normalizes features to zero mean and unit variance so no single feature dominates due to scale differences
- **Stratified Split** — Ensures both train and test sets maintain the same class ratio (62% benign / 38% malignant) as the original dataset
- **Dropout** — Regularization technique that prevents overfitting by randomly deactivating neurons during training
- **Gradient Clipping** — Prevents exploding gradients by capping gradient norm at 1.0
- **ReduceLROnPlateau** — Reduces learning rate when validation loss stops improving, helping fine-tune convergence

---

## 👤 Author

**Your Name**
- GitHub: [@tafsir99-gg](https://github.com/tafsir99-gg)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
