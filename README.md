# 🛡️ SMS Spam Detector

> An AI-powered web application that detects SMS spam messages in real time using a custom Transformer deep learning model built from scratch with PyTorch and deployed with Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smsspamtransformermodel-y3.streamlit.app)

---

## 🚀 Live Demo

**👉 [https://smsspamtransformermodel-y3.streamlit.app](https://smsspamtransformermodel-y3.streamlit.app)**

---

## ✨ Features

- **Single Message Detection** — Paste any SMS and instantly get SPAM or HAM verdict with confidence score, spam probability, and risk level
- **Spam Signal Indicators** — Highlights detected spam signals (FREE, WIN, URGENT, CLICK, money amounts, URLs, etc.)
- **Batch Processing** — Upload a `.txt` file with multiple messages and get a full breakdown with stats
- **Report Generation** — Download a detailed spam detection report for uploaded files
- **Model Statistics** — View accuracy, precision, recall, F1-score, architecture details, and baseline comparison

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.92% |
| Precision | 97.81% |
| Recall    | 94.51% |
| F1-Score  | 96.13% |

---

## 🏗️ Model Architecture

```
Input SMS Message
       ↓
Text Cleaning + Tokenization + Padding (length = 150)
       ↓
GloVe 300d Word Embeddings + Positional Encoding
       ↓
Encoder × 3  (Multi-Head Self-Attention + Feed Forward)
       ↓
Decoder × 3  (Cross-Attention + Learnable Memory Vectors)
       ↓
Linear Layers  (256 → 128 → 64 → 1)
       ↓
Sigmoid → Probability → Threshold → SPAM / HAM
```

---

## 📁 Project Structure

```
SMS_SpamDetector/
├── streamlit_app.py       ← Main Streamlit app (entry point)
├── requirements.txt       ← Python dependencies
├── README.md
├── src/
│   ├── config.py          ← Hyperparameters and settings
│   ├── model.py           ← Transformer architecture
│   ├── preprocess.py      ← Text cleaning and tokenization
│   ├── predict.py         ← Prediction logic
│   └── train.py           ← Model training loop
├── models/
│   ├── spam_transformer.pth  ← Trained model weights
│   └── vocab.pkl             ← Vocabulary mappings
├── data/
│   └── spam.csv           ← Training dataset (5,572 messages)
└── docs/
    └── data_preprocessing_diagram.png
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/SMS_SpamDetector.git
cd SMS_SpamDetector
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

---

## 🔧 Train the Model (Optional)

If you want to retrain the model on your own:

```bash
cd src
python train.py
```

This will:
- Load and preprocess `data/spam.csv`
- Train the Transformer for up to 50 epochs with early stopping
- Auto-tune the best classification threshold on validation set
- Save best weights to `models/spam_transformer.pth`
- Save vocabulary to `models/vocab.pkl`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch |
| NLP | Custom Tokenizer + GloVe 300d |
| Model | Custom Transformer (Encoder-Decoder) |
| Frontend | Streamlit |
| Language | Python 3.10 |
| ML Utils | Scikit-learn, NumPy, Pandas |

---

## 📦 Requirements

```
streamlit
torch
scikit-learn
pandas
numpy
tqdm
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

Uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) dataset — 5,572 labeled SMS messages (spam + ham).

---

## 👨‍💻 Authors

**S. Sathyasai** — [23B81A67A6]

**T. Srikar Reddy** — [23B81A67B3]

CVR College of Engineering · Department of Computer Science and Engineering

---

## 📄 License

This project is built for academic purposes at CVR College of Engineering.
