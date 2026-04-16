# 🔍 Industrial Defect Inspector

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![FastViT](https://img.shields.io/badge/Model-FastViT--T8-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

An end-to-end industrial anomaly detection system that classifies manufacturing parts as **Good** or **Defective** in real-time, using **FastViT** — a fast, accurate Vision Transformer architecture optimised for edge deployment.

> Built to simulate a real factory inspection line, targeting deployment on resource-constrained hardware via C++ and ONNX.

---

## 🧠 The Problem

The [MVTec AD dataset](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) is typically used for *unsupervised* anomaly detection. This project reframes it as a **binary classification task** (`good` vs `damaged`) to evaluate FastViT's performance in a *data-scarce* industrial environment — where defective samples are limited and misclassification has real-world cost.

The core challenge: train a model that can spot tiny scratches or cracks on a live production line, fast enough to keep up with the belt.

---

## 🗺️ Project Roadmap

### ✅ Phase 1 — The Intelligence *(Complete)*
- Dataset: MVTec AD (Bottle category), restructured into binary classification
- Model: **FastViT-T8** via the `timm` library, fine-tuned from ImageNet weights
- Tackled class imbalance using **weighted CrossEntropyLoss** (~3.6× weight on damaged class)
- Trained in Google Colab with a 70/15/15 train/val/test split
- Achieved strong validation accuracy across 10 epochs

### 🔄 Phase 2 — The Software *(In Progress)*
- Migrate from Jupyter Notebook to clean `.py` scripts
- Build a **simulated video stream** using OpenCV — reads images from a folder one-by-one to replicate a factory camera feed
- Real-time prediction overlay on each frame

### 📦 Phase 3 — The Packaging *(Planned)*
- Wrap the inference pipeline in a **Docker container**
- Ensures the system runs identically on any machine — no environment issues
- Demonstrates core **MLOps** practices relevant to production deployment

### ⚡ Phase 4 — The Speed *(Planned)*
- Convert the trained PyTorch model to **ONNX** format
- Write a **C++ inference program** using ONNX Runtime
- Target: edge AI deployment on hardware where Python is too slow (e.g. embedded systems, Raspberry Pi, Jetson Nano)

---

## 🏗️ Project Structure

```
industrial-defect-inspector/
│
├── notebooks/
│   └── defect_analysis.ipynb      # Full walkthrough (Colab-ready)
│
├── src/
│   ├── data_preprocessing.py      # Data gathering, splitting & DataLoaders
│   ├── model.py                   # FastViT setup, training & validation loops
│   └── utils.py                   # Save/load weights, inference, evaluation
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Tech Stack

| Area | Tool |
|---|---|
| Model | FastViT-T8 (`timm`) |
| Framework | PyTorch + torchvision |
| Dataset | MVTec AD |
| Video Stream | OpenCV *(Phase 2)* |
| Containerisation | Docker *(Phase 3)* |
| Export | ONNX *(Phase 4)* |
| Edge Inference | C++ + ONNX Runtime *(Phase 4)* |
| Training | Google Colab |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Katherinejimmy/industrial-defect-inspector-vit.git
cd industrial-defect-inspector-vit
pip install -r requirements.txt
```

### 2. Download the dataset

```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/ipythonx/mvtec-ad?select=bottle")
```

### 3. Build the dataset splits

```python
from src.data_preprocessing import build_dataset, get_dataloaders

build_dataset(
    mvtec_root="/content/mvtec-ad/bottle",
    base_dir="/content/industrial_data"
)

train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
    base_dir="/content/industrial_data"
)
```

### 4. Train

```python
from src.model import build_model, get_criterion_and_optimizer, train

model, device = build_model()
criterion, optimizer = get_criterion_and_optimizer(model, device)
history = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
```

### 5. Evaluate & infer

```python
from src.utils import evaluate, predict_image, save_model

evaluate(model, val_loader, train_dataset.classes, device)
predict_image(model, "path/to/image.png", train_dataset.classes, device)
save_model(model, "fastvit_bottle_inspector.pth")
```

For the full walkthrough, open [`notebooks/defect_analysis.ipynb`](notebooks/defect_analysis.ipynb) in Google Colab.

---

## 📊 Results (Phase 1)

| Metric | Value |
|---|---|
| Architecture | FastViT-T8 |
| Epochs | 10 |
| Optimiser | AdamW (lr=1e-4) |
| Loss | Weighted CrossEntropyLoss |
| Val Accuracy | ~95%+ |

---

## 📌 Why FastViT?

Standard Vision Transformers (ViT) are powerful but slow — too slow for real-time factory inspection. FastViT uses a hybrid token mixing strategy that achieves **comparable accuracy at significantly lower latency**, making it a strong candidate for the C++ edge deployment targeted in Phase 4.

---

## 📄 License

MIT License. The MVTec AD dataset has its own [licence](https://www.mvtec.com/company/research/datasets/mvtec-ad) — please review before use.
