# SMS Spam Detector

A web application that detects spam SMS messages using a custom-built Transformer deep learning model. Built from scratch with PyTorch, Flask, and vanilla JavaScript.

---

## What it does

- Paste any SMS message and instantly know if it's **Spam or Ham**
- Shows confidence score and detected spam signals
- Background flashes **red** for spam, **green** for ham
- Upload a `.txt` file to analyze multiple messages at once
- Download a full detection report

---

## Project Structure

```
SMS-Spam-Detector/
├── frontend/
│   ├── index.html       # UI
│   ├── style.css        # Dark theme styling
│   └── script.js        # API calls and animations
├── backend/
│   ├── model.py         # Transformer architecture
│   ├── train.py         # Training loop
│   ├── predict.py       # Inference logic
│   ├── preprocess.py    # Text cleaning and tokenization
│   ├── app.py           # Flask API server
│   └── config.py        # Hyperparameters
├── data/
│   └── sms_spam.csv     # Training dataset (5771 messages)
├── run.py               # Main entry point
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
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

### 4. Train the model

```bash
python run.py --train
```

This will:
- Load and preprocess the dataset
- Train the Transformer model for up to 50 epochs
- Save the best model to `backend/models/spam_transformer.pth`
- Save the vocabulary to `backend/vocab.pkl`

Training takes around 5–15 minutes depending on your machine.

### 5. Start the server

```bash
python backend/app.py
```

### 6. Open the app

Go to `http://localhost:5000` in your browser.

---

## Model Architecture

```
Input SMS
    ↓
Word Embedding + Positional Encoding
    ↓
Encoder × 3  (Self-Attention + FeedForward)
    ↓
Decoder × 3  (Cross-Attention with Memory Vectors)
    ↓
Linear Layers → Sigmoid
    ↓
Spam or Ham
```

---

## Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.92% |
| Precision | 0.9781 |
| Recall    | 0.9451 |
| F1-Score  | 0.9613 |

---

## Requirements

- Python 3.8+
- PyTorch
- Flask
- scikit-learn
- pandas
- numpy
- tqdm

All listed in `requirements.txt`.

---

## Dataset

Uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) dataset with additional tricky spam/ham examples added for better generalization.

---

## Author

Sathyasai

Srikar
