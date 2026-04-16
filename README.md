# Industrial Defect Inspector

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![FastViT](https://img.shields.io/badge/Model-FastViT--T8-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A system that watches a live stream of industrial parts and flags them as **Good** or **Defective** in real time. The end goal is running inference in C++ on edge hardware.

---

## The idea

MVTec AD is normally used for unsupervised anomaly detection. I restructured it into a binary classification task instead, partly to see how FastViT handles a data-scarce setup (defective samples are always the minority in real factories)

The four-phase plan below is what I mapped out before starting. Phase 1 is done.

---

## Roadmap

**Phase 1 — Train the model (done)**

The first problem was the dataset itself. MVTec's train split only contains `good` images, there are no defects in it at all. So before any training could happen, I had to pull the damaged samples out of the test folder (broken large, broken small, contamination, etc.), pool them with the good images, and rebuild the whole thing into a proper `train/val/test` structure at 70/15/15.

That left a second problem: 209 good images vs 63 damaged. To stop the model from just predicting "good" for everything, I used weighted CrossEntropyLoss and gave the damaged class 3.6× the weight of good, calculated as `total_samples / (num_classes × samples_in_class)`.

- Model: FastViT-T8 via `timm`, fine-tuned from ImageNet weights
- Optimiser: AdamW (lr=1e-4), 10 epochs on Google Colab
- ~95%+ validation accuracy

**Phase 2 — Simulated video stream (in progress)**
- Move from notebook to `.py` scripts
- OpenCV loop that reads images from a folder one-by-one to simulate a factory camera feed
- Prediction label overlaid on each frame

**Phase 3 — Docker**
- Wrap the whole pipeline in a container
- The point: it should run identically on any machine, no environment excuses, this is what actually matters in a production setting

**Phase 4 — C++ edge deployment**
- Export the trained model to ONNX
- Write a C++ inference program using ONNX Runtime
- Target hardware where Python is too slow: Raspberry Pi, Jetson Nano, that kind of thing

---

## Project structure

```
industrial-defect-inspector/
│
├── notebooks/
│   └── defect_analysis.ipynb      # Full walkthrough, Colab-ready
│
├── src/
│   ├── data_preprocessing.py      # Dataset restructuring and DataLoaders
│   ├── model.py                   # FastViT setup, training and validation
│   └── utils.py                   # Save/load, inference, evaluation
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech stack

| Area | Tool |
|---|---|
| Model | FastViT-T8 (`timm`) |
| Framework | PyTorch + torchvision |
| Dataset | MVTec AD |
| Video stream | OpenCV (Phase 2) |
| Containerisation | Docker (Phase 3) |
| Export | ONNX (Phase 4) |
| Edge inference | C++ + ONNX Runtime (Phase 4) |
| Training | Google Colab |

---

## Quick start

### 1. Clone and install

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

### 3. Build dataset splits

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

### 5. Evaluate and run inference

```python
from src.utils import evaluate, predict_image, save_model

evaluate(model, val_loader, train_dataset.classes, device)
predict_image(model, "path/to/image.png", train_dataset.classes, device)
save_model(model, "fastvit_bottle_inspector.pth")
```

Full walkthrough: [`notebooks/defect_analysis.ipynb`](Notebooks/Industrial_Defect_Inspector_pipeline.ipynb) — open in Colab.

---

## Results (Phase 1)

| Metric | Value |
|---|---|
| Architecture | FastViT-T8 |
| Epochs | 10 |
| Optimiser | AdamW (lr=1e-4) |
| Loss | Weighted CrossEntropyLoss (damaged: 3.6×) |
| Val Accuracy | ~95%+ |

---

## License

MIT. The MVTec AD dataset has its own [licence](https://www.mvtec.com/company/research/datasets/mvtec-ad) 
