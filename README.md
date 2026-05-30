# Pneumonia_detector
# 🫁 PneumoScan AI

A deep learning-powered chest X-ray analysis web application built with **Streamlit** and **PyTorch**. It classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a fine-tuned ResNet18 model trained via transfer learning.

---

## 📸 Features

- **Single Scan** — Upload one X-ray and get instant prediction with confidence score, probability bars, and a risk meter
- **Batch Scan** — Upload multiple X-rays at once; get a summary dashboard and download results as CSV
- **Image Processing Modes** — View scans in Original, High Contrast, Sharpen, Edge Highlight, or Equalized mode
- **Severity Rating** — LOW / MODERATE / HIGH / CRITICAL classification based on pneumonia probability
- **Clinical Notes** — Contextual interpretation and recommended next steps for each result
- **Model Info Sidebar** — Live display of test accuracy, AUC-ROC, epochs trained, and system status
- **Sci-fi Dark UI** — Custom CSS with Orbitron/Exo 2 fonts, neon accents, animated risk meter

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet18 (pretrained on ImageNet) |
| Task | Binary Classification — NORMAL vs PNEUMONIA |
| Trainable Parameters | ~66K (custom head only) |
| Input Size | 224 × 224 RGB |
| Normalization | ImageNet mean/std |
| Output | Softmax probabilities for 2 classes |

### Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Regularization | Dropout(0.4) + Dropout(0.3) |
| Class Imbalance | WeightedRandomSampler |
| Early Stopping | Patience = 4 epochs on val loss |
| Mixed Precision | float16 (GPU only) |
| Max Epochs | 15 |

### Model Architecture (Custom Head)

```
ResNet18 (frozen backbone)
    └── Dropout(p=0.4)
    └── Linear(512 → 256)
    └── ReLU()
    └── Dropout(p=0.3)
    └── Linear(256 → 2)
```

### Expected Performance

| Metric | Value |
|---|---|
| Test Accuracy | ~90–94% |
| AUC-ROC | > 0.95 |

---

## 📁 Project Structure

```
pneumoscan/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── pneumonia_model.pkl       # Trained model (download from Colab)
├── model_metadata.json       # Class names, accuracy, input specs
├── training_history.pkl      # Loss/accuracy per epoch (optional)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/pneumoscan-ai.git
cd pneumoscan-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model files

Download the following files from your Google Colab training notebook and place them in the project root:

- `pneumonia_model.pkl`
- `model_metadata.json`
- `training_history.pkl` *(optional)*

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit>=1.35.0
torch>=2.2.0
torchvision>=0.17.0
Pillow>=10.0.0
numpy>=1.24.0
```

---

## 🗂️ Dataset

The model was trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

```
chest_xray/
├── train/
│   ├── NORMAL/        (1,341 images)
│   └── PNEUMONIA/     (3,875 images)
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/        (234 images)
    └── PNEUMONIA/     (390 images)
```

> The training notebook fixes the tiny val set (16 images → ~312) by moving 50% of test images into val for stable validation curves.

---

## 🖥️ Usage Guide

### Single Scan (SCAN tab)

1. Open the app in your browser
2. Select an image enhancement mode from the sidebar (optional)
3. Upload a chest X-ray image (JPG/PNG)
4. View the prediction result, confidence score, probability bars, and risk meter
5. Expand **Clinical Notes** for interpretation and recommendations

### Batch Scan (BATCH tab)

1. Switch to the **BATCH** tab
2. Upload multiple chest X-ray images at once
3. The app scans all images and shows a summary (total, normal, pneumonia, detection rate)
4. Browse individual results in a 3-column grid
5. Click **Download CSV Report** to export results

### Severity Levels

| Level | Pneumonia Probability |
|---|---|
| 🟢 LOW | < 30% |
| 🟡 MODERATE | 30–60% |
| 🟠 HIGH | 60–80% |
| 🔴 CRITICAL | > 80% |

---

## 🔄 Retraining the Model

Open `chest_xray_final.ipynb` in Google Colab:

1. Go to **Runtime → Change runtime type → T4 GPU**
2. Mount your Google Drive containing the dataset
3. Run all cells — the notebook auto-detects the dataset path
4. Download `pneumonia_model.pkl`, `model_metadata.json`, and `training_history.pkl`
5. Replace the files in your app folder

---

## ⚠️ Disclaimer

This application is intended **strictly for research and educational purposes**. It is **not** a certified medical device and should **not** be used for clinical diagnosis or patient care decisions. Always consult a licensed physician or radiologist for medical evaluation.

---

## 🛠️ Tech Stack

- [PyTorch](https://pytorch.org/) — model training and inference
- [Torchvision](https://pytorch.org/vision/) — ResNet18, transforms
- [Streamlit](https://streamlit.io/) — web application framework
- [Pillow](https://python-pillow.org/) — image loading and processing
- [scikit-learn](https://scikit-learn.org/) — evaluation metrics (AUC-ROC, confusion matrix)

---

*Built with ❤️ using Transfer Learning on ResNet18*
