import torch
import pandas as pd
import nltk
from collections import Counter
import numpy as np
from model import SpamTransformer
import joblib
import os

nltk.download('punkt', quiet=True)

def load_model(vocab_path='models/vocab.pkl', model_path='models/spam_transformer.pth'):
    with open(vocab_path, 'rb') as f:
        vocab = joblib.load(f)
    model = SpamTransformer(len(vocab))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, vocab

def preprocess_text(text, vocab, max_len=100):
    tokens = nltk.word_tokenize(text.lower())
    seq = [vocab.get(t, vocab['<unk>']) for t in tokens[:max_len]]
    if len(seq) < max_len:
        seq += [vocab['<pad>']] * (max_len - len(seq))
    return torch.tensor([seq])

def predict(model, vocab, text):
    with torch.no_grad():
        input_tensor = preprocess_text(text, vocab)
        prob = torch.sigmoid(model(input_tensor)).item()
        return 'spam' if prob > 0.5 else 'ham', prob

def process_file(file, model, vocab):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
        if 'sms_message' not in df.columns:
            raise ValueError("CSV must have 'sms_message' column")
        messages = df['sms_message'].astype(str).values
    else:  # TXT, one message per line
        messages = pd.read_csv(file, header=None)[0].astype(str).values
    
    results = []
    spam_count, ham_count = 0, 0
    for msg in messages:
        pred, prob = predict(model, vocab, msg)
        results.append({'original': msg, 'prediction': pred, 'probability': prob})
        if pred == 'spam':
            spam_count += 1
        else:
            ham_count += 1
    
    # Regenerate cleaned file: keep only ham or flag spam
    cleaned_df = pd.DataFrame(results)
    output_path = f"processed_files/processed_{os.path.basename(file.name)}"
    cleaned_df.to_csv(output_path, index=False)
    
    return spam_count, ham_count, len(messages), output_path
