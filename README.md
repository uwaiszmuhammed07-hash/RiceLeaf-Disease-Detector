# 🌾 RiceGuard AI — Rice Leaf Disease Detection

> An end-to-end deep learning project that detects rice leaf diseases from images using InceptionV3 Transfer Learning, deployed as a live web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://riceguard-ai-l5qwxixmavk5h8pcffz9g2.streamlit.app/)

---

## 🎯 Problem Statement

Rice is one of the most important food crops in the world. Disease in rice leaves can significantly reduce crop yield. This project aims to automatically classify rice leaf diseases into one of three categories using deep learning.

**Tasks:**
- Task 1 → Complete data analysis report on the dataset
- Task 2 → Build a model to classify three rice leaf diseases
- Task 3 → Analyze augmentation techniques and create a report

---

## 🌿 Detectable Diseases

| Disease | Description |
|---------|-------------|
| 🟡 Bacterial Leaf Blight | Yellow/white lesions along leaf edges |
| 🟤 Brown Spot | Oval brown spots scattered on leaf |
| ⚫ Leaf Smut | Dark raised spots on leaf surface |

---

## 📊 Dataset

- **Total Images:** 119 JPG images
- **Classes:** 3 (Bacterial Leaf Blight, Brown Spot, Leaf Smut)
- **Split:** 70% Train | 15% Validation | 15% Test
- **Source:** [Download Dataset](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1001-RiceLeaf.zip)

---

## 🏆 Model Comparison

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| Custom CNN | 33.33% | Baseline — insufficient data |
| ResNet50 | 44.44% | BatchNorm conflict |
| VGG16 | 66.67% | Unstable on small dataset |
| MobileNetV2 | 83.33% | Stable training |
| **InceptionV3** | **94.44%** | **🏆 Best Model** |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Models:** InceptionV3, MobileNetV2, VGG16, ResNet50
- **UI:** Streamlit
- **Deployment:** Streamlit Cloud
- **Version Control:** GitHub + Git LFS

---

## 🚀 How to Run Locally

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Shihadkp/riceguard-ai.git
cd riceguard-ai
```

### Step 2 — Create Virtual Environment
```bash
conda create -n riceguard python=3.10 -y
conda activate riceguard
```

### Step 3 — Install Requirements
```bash
pip install -r requirements.txt
```

### Step 4 — Run the App
```bash
streamlit run app.py
```

### Step 5 — Open in Browser
```
http://localhost:8501
```

---

## 🌐 Live Demo

The app is deployed on Streamlit Cloud:

👉 [Open RiceGuard AI](https://shihadkp-riceguard-ai-app.streamlit.app)

---

## 📁 Project Structure

```
riceguard-ai/
├── app.py                          ← Streamlit web app
├── requirements.txt                ← Dependencies
├── saved_models/
│   ├── best_model_inceptionv3.keras ← Best trained model
│   └── class_labels.json           ← Class label mapping
└── README.md
```

---

## 🔬 App Features

- 📤 Upload any rice leaf image
- 🔍 Instant disease prediction
- 📊 Confidence scores for all 3 classes
- 📋 Disease description & treatment info
- 🚫 Invalid image detection (non-leaf rejection)
- 🎨 Beautiful dark themed UI

---

## 📈 Key Findings

- Transfer Learning improved accuracy by **61%** over Custom CNN
- InceptionV3's multi-scale Inception modules best suited for disease pattern detection
- Fine-tuning did not improve results due to small dataset size (119 images)
- Collecting 500+ images per class could push accuracy to 97-100%

---

## ⚠️ Challenges Faced

1. **Small Dataset** → Solved with 8 augmentation techniques
2. **Variable Image Sizes** → Automatic resizing to 299×299
3. **ResNet50 Failure** → BatchNorm conflict with frozen layers
4. **Fine-tuning Backfired** → Restored pre-finetune checkpoint
5. **Class Similarity** → InceptionV3 multi-scale detection solved this

---

## 📋 Requirements

```
tensorflow==2.20.0
streamlit
numpy
opencv-python-headless
Pillow
matplotlib
```

---

## 👨‍💻 Author

**Muhammed Shihad**

[![GitHub](https://img.shields.io/badge/GitHub-Shihadkp-black?logo=github)](https://github.com/Shihadkp)

---

**Project ID:** PRCP-1001 | **Model:** InceptionV3 | **Accuracy:** 94.44%
