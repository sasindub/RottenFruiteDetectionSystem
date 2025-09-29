# FreshGuard AI — Rotten Fruit Detection (Apples, Bananas, Oranges)

An end‑to‑end fruit quality detection system that classifies images as fresh or rotten for apples, bananas, and oranges. It includes:

- Model training scripts (`model.py`) that train a Simple CNN and a MobileNetV2 transfer learning model
- Saved models (`simple_cnn_model.h5`, `mobilenetv2_model.h5`, `best_model.h5`)
- A FastAPI web app (`main.py`) with a modern UI (`templates/index.html`) for single‑image upload and prediction

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset Layout](#dataset-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Train Models](#train-models)
- [Run the Web App](#run-the-web-app)
- [Models and Accuracy](#models-and-accuracy)
- [Notes and Tips](#notes-and-tips)
- [License](#license)
- [Author](#author)

---

## Features
- **Two models**: Simple CNN and MobileNetV2 (transfer learning)
- **Automatic best model selection** saved as `best_model.h5`
- **FastAPI + Jinja2 UI**: drag‑and‑drop upload, live preview, styled results with confidence bar
- **Classes**: `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`

---

## Project Structure
```
RottenFruiteDetectionSystem/
├─ dataset/
├─ main.py                 # FastAPI app (serves UI, /predict endpoint)
├─ model.py                # Training for Simple CNN & MobileNetV2
├─ templates/
│  └─ index.html           # Modern single‑page UI
├─ simple_cnn_model.h5     # Saved model after training
├─ mobilenetv2_model.h5    # Saved model after training
├─ best_model.h5           # Best of the two based on val_accuracy
└─ README.md
```

---

## Dataset Layout
```
dataset/
├─ Train/
│  ├─ freshapples/
│  ├─ rottenapples/
│  ├─ freshbanana/
│  ├─ rottenbanana/
│  ├─ freshoranges/
│  └─ rottenoranges/
└─ Test/
   ├─ freshapples/
   ├─ rottenapples/
   ├─ freshbanana/
   ├─ rottenbanana/
   ├─ freshoranges/
   └─ rottenoranges/
```

- Images can be JPG/PNG/WebP.
- Class folder names must match the above exactly; training code infers labels from folders.

---

## Requirements
- Python 3.10+
- Recommended GPU with CUDA for faster training (CPU works but is slower)

Python packages (installed below): TensorFlow, scikit‑learn, FastAPI, Uvicorn, Pillow, NumPy.

---

## Setup
On Windows PowerShell:

```powershell
# 1) Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate

# 2) Install dependencies
pip install --upgrade pip
pip install tensorflow fastapi uvicorn[standard] pillow numpy scikit-learn jinja2
```

Place the `dataset/` directory at the project root matching the [Dataset Layout](#dataset-layout).

---

## Train Models
`model.py` trains two models and saves:
- `simple_cnn_model.h5`
- `mobilenetv2_model.h5`
- `best_model.h5` (whichever achieved higher validation accuracy)

Run:
```powershell
python model.py
```
Key defaults inside `model.py`:
- Image size: 128×128
- Batch size: 16
- Epochs: 10

---

## Run the Web App
The app loads `best_model.h5` and serves a web UI.

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Then open: `http://127.0.0.1:8000`

Usage:
- Upload a single fruit image (apple, banana, or orange)
- Get prediction such as `freshbanana` with confidence percentage

---

## Models and Accuracy
- Simple CNN: 3 conv blocks, dropout, dense head; trained from scratch
- MobileNetV2: ImageNet‑pretrained backbone, frozen for feature extraction + GAP + dense head
- The UI highlights a nominal accuracy of ~98% shown in the page; actual accuracy depends on your dataset split and size. Refer to the printed `classification_report` in console after training for precise metrics.

---

## Notes and Tips
- If you already have `best_model.h5`, you can skip training and run the app directly.
- Ensure class names in `main.py` match your dataset folders if you customize classes.
- If you change image size or classes, retrain and regenerate `best_model.h5`.
- For larger datasets, consider enabling GPU and increasing epochs.

---

## License
This project is for academic purposes. No commercial use allowed.

## Author
Sasindu Bandara — BSc Computing Student, Sri Lanka
