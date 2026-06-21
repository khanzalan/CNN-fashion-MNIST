# 👗 Fashion-MNIST Image Classification — CNN

A deep learning project that trains a **Convolutional Neural Network (CNN)** to classify 10 clothing categories from the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset using TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~91%25-brightgreen)

---

## 📌 Project Overview

Fashion-MNIST is a drop-in replacement for MNIST, with **70,000 grayscale clothing images** (28×28 px) across 10 classes. This project demonstrates why CNNs outperform flat ANNs on image data, using BatchNormalization, Dropout, L2 regularization, and EarlyStopping to build a generalized classifier.

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Source | `keras.datasets.fashion_mnist` |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 × 28 pixels, grayscale |
| Classes | 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |

---

## 🏗️ Model Architecture

```
Input (28×28×1)
    │
    ├─ Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.20)
    │           Output: 14×14×32
    │
    ├─ Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
    │           Output: 7×7×64
    │
    ├─ Block 3: Conv2D(128) → BatchNorm → ReLU → Dropout(0.30)
    │           Output: 7×7×128
    │
    └─ Head   : Flatten → Dense(128, L2) → Dropout(0.5)
                        → Dense(64,  L2) → Dropout(0.2)
                        → Dense(10, Softmax)
```

- **Loss:** Sparse Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Epochs:** up to 30 with EarlyStopping (patience=7)  
- **Batch size:** 64

---

## 📈 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~91% |
| Test Loss | ~0.26 |

### Training Curves
![Training Curves](training_curves.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Sample Predictions
![Sample Predictions](sample_predictions.png)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fashion-mnist-cnn.git
cd fashion-mnist-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Fashion_MNIST_CNN.ipynb
```

The dataset downloads automatically on first run via `keras.datasets`.

---

## 📁 Project Structure

```
fashion-mnist-cnn/
├── Fashion_MNIST_CNN.ipynb     # Main notebook
├── requirements.txt             # Dependencies
├── fashion_cnn_model.keras      # Saved model (generated after training)
├── sample_images.png            # Generated visualizations
├── training_curves.png
├── confusion_matrix.png
├── sample_predictions.png
└── README.md
```

---

## 🔑 Key Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Spatial feature extraction | `Conv2D` layers |
| Dimensionality reduction | `MaxPooling2D` |
| Training stability | `BatchNormalization` |
| Overfitting prevention | `Dropout` + L2 regularization |
| Auto-stopping | `EarlyStopping` callback |
| Model persistence | `model.save()` + `load_model()` |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow 2.x / Keras** — CNN model
- **scikit-learn** — Confusion matrix
- **NumPy** — Array operations
- **Matplotlib** — Visualizations
