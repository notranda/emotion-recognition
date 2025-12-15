# 🎭 Emotion Recognition System (MLOps Project)

## 📌 Overview

**Emotion Recognition System** adalah proyek *Machine Learning Operations (MLOps)* yang bertujuan untuk mengklasifikasikan emosi manusia berdasarkan gambar wajah. Proyek ini dibangun secara *end-to-end*, mulai dari training model *Deep Learning*, versioning model, retraining, hingga deployment ke aplikasi **Streamlit**.

Sistem ini mendukung **dua arsitektur model**:

* **CNN (Convolutional Neural Network)** – model ringan dan efisien
* **ResNet50** – model *transfer learning* dengan performa lebih kuat

Aplikasi memungkinkan pengguna mengunggah gambar wajah, kemudian sistem akan memprediksi emosi secara otomatis.

---

## 🎯 Objectives

* Membangun sistem klasifikasi emosi wajah berbasis Deep Learning
* Mengimplementasikan konsep **MLOps** (training, retraining, versioning, deployment)
* Menyediakan aplikasi web interaktif menggunakan **Streamlit**
* Mendukung pemilihan model secara dinamis (CNN / ResNet)

---

## 🧠 Emotions Classes

Model dikembangkan untuk mengenali **7 kelas emosi**:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😄
* Neutral 😐
* Sad 😢
* Surprise 😲

---

## 📂 Project Structure

```bash
emotion-recognition/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Dependencies
├── .gitignore
│
├── mlops/
│   ├── cli.py                 # CLI for training & retraining
│   └── infer.py               # Inference utilities
│
├── services/
│   └── retraining_service.py  # Training & retraining logic
│
├── models/
│   ├── README.md              # Model directory description
│   └── active/
│       ├── cnn/
│       │   ├── model.keras
│       │   └── metadata.json
│       └── resnet/
│           ├── model.keras
│           └── metadata.json
│
├── data/
│   └── README.md              # Dataset description
│
└── docs/                      # Documentation & assets
```

---

## 📊 Dataset

Dataset yang digunakan adalah **FER2013**, yang berisi gambar wajah berukuran **48x48 grayscale**.

**Catatan:**

* Dataset **tidak disertakan** di dalam repository
* Dataset dapat diperoleh dari Kaggle: *FER2013 Emotion Dataset*

---

## 🏗 Model Architecture

### 1️⃣ CNN Model

* Input: `48x48x1` (grayscale)
* Conv2D + BatchNorm + MaxPooling
* Fully Connected Layers
* Output: Softmax (7 kelas)

Model ini ringan dan cocok untuk deployment cepat.

---

### 2️⃣ ResNet50 Model

* Transfer Learning (ImageNet weights)
* Input: `48x48x3`
* Freeze sebagian layer
* Fully Connected head

Model ini memiliki performa lebih tinggi namun lebih berat.

---

## 🔄 MLOps Workflow

```text
Data → Training → Evaluation → Model Versioning → Deployment → Inference
                        ↑
                   Retraining
```

Fitur MLOps yang diimplementasikan:

* Model versioning
* Retraining pipeline
* Metadata tracking (model, versi, tanggal, akurasi)

---

## 🚀 Running the Application

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🖼 Streamlit Features

* Upload image (RGB → otomatis dikonversi ke grayscale untuk CNN)
* Pilih model (CNN / ResNet50)
* Tampilkan probabilitas tiap emosi
* Informasi model (versi, arsitektur)

---

## 🧪 CLI Training & Retraining

```bash
python -m mlops.cli --model cnn --epochs 10
python -m mlops.cli --model resnet --epochs 10
```

---

## 🧪 Experiments & Evaluation

### 🔍 Experimental Setup

Eksperimen dilakukan untuk membandingkan performa dua arsitektur model:

| Model    | Input Size | Color Mode                       | Epochs | Optimizer |
| -------- | ---------- | -------------------------------- | ------ | --------- |
| CNN      | 48×48      | Grayscale                        | 10–60  | Adam      |
| ResNet50 | 48×48      | RGB (from grayscale replication) | 10–60  | Adam      |

Dataset dibagi menjadi:

* **Training set**: 80%
* **Validation set**: 20%

Augmentasi data diterapkan pada training set untuk meningkatkan generalisasi model.

---

### 📈 Evaluation Metrics

Model dievaluasi menggunakan metrik berikut:

* **Accuracy**
* **Categorical Cross-Entropy Loss**

Evaluasi dilakukan pada validation set selama proses training.

---

### 📊 Experimental Results (Summary)

| Model    | Validation Accuracy (±) | Notes                     |
| -------- | ----------------------- | ------------------------- |
| CNN      | ~65–70%                 | Ringan, cepat, stabil     |
| ResNet50 | ~70–75%                 | Lebih akurat, lebih berat |

> *Catatan: Nilai akurasi dapat berbeda tergantung jumlah epoch, augmentasi, dan distribusi data.*

---

### 🧠 Analysis

* CNN cocok untuk deployment cepat dengan resource terbatas
* ResNet50 menunjukkan performa lebih baik dalam mengenali ekspresi kompleks
* Konversi grayscale → RGB tetap memungkinkan transfer learning bekerja efektif

---

### ⚠ Threats to Validity

* Dataset FER2013 memiliki ketidakseimbangan kelas
* Ekspresi wajah ambigu dapat menyebabkan mis-klasifikasi
* Kualitas gambar (noise, pencahayaan) memengaruhi prediksi

bash
python -m mlops.cli --model cnn --epochs 10
python -m mlops.cli --model resnet --epochs 10

```

---

## ⚠ Limitations
- Tidak mendukung real-time webcam (Streamlit limitation)
- Akurasi bergantung pada kualitas gambar wajah

---

## 🛠 Technologies Used
- Python 3.10+
- TensorFlow / Keras
- OpenCV
- Streamlit
- Typer (CLI)
- NumPy, Pandas

---

## 📜 License
This project is licensed under the **MIT License**.

You are free to:
- Use
- Modify
- Distribute

with proper attribution.

---

## 👤 Team
**Randa Andriana Putra**  
**Residen Nusantara R.M**
**Kayla Amanda Sukma**
**Farahanum Afifah A**
**Safitri** 

---

## ⭐ Acknowledgments
- Kaggle FER2013 Dataset
- TensorFlow & Streamlit Community
- Academic MLOps references

---

> *This project was developed for academic purposes as part of an MLOps course.*

```
