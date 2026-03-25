"""
Data preprocessing module
Handles tokenization, embedding, and data loading
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os
import re
from sklearn.model_selection import train_test_split

class SMSDataset(Dataset):
    """Custom Dataset class for SMS messages"""
    
    def __init__(self, texts, labels, word2idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Tokenize and convert to indices
        tokens = self.texts[idx].split()
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens[:self.max_length]]
        
        # Padding
        padding_length = self.max_length - len(indices)
        indices = indices + [self.word2idx['<PAD>']] * padding_length
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, max_length=150, vocab_size=50000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.word2idx = None
        self.idx2word = None
        self.vocab = None
        
    def clean_text(self, text):
        if isinstance(text, str):
            # Lowercase
            text = text.lower()
            
            # Replace URLs with token
            text = re.sub(r'http\S+|www\S+|https\S+', ' <URL> ', text, flags=re.MULTILINE)
            
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Keep letters, digits, $, !, ?, % — strong spam signals
            text = re.sub(r'[^a-zA-Z0-9\s\$\!\?\%]', ' ', text)
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        return ""
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts
        Creates word to index mapping
        """
        word_counts = Counter()
        
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(self.vocab_size - 3)  # Reserve for special tokens
        
        # Create vocabulary
        self.word2idx = {
            '<PAD>': 0,   # Padding token
            '<UNK>': 1,   # Unknown token
            '<START>': 2  # Start token
        }
        
        self.idx2word = {
            0: '<PAD>',
            1: '<UNK>',
            2: '<START>'
        }
        
        # Add words to vocabulary
        for idx, (word, _) in enumerate(most_common, start=3):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.vocab = self.word2idx
        
        return self.word2idx
    
    def load_and_preprocess_data(self, csv_path):
        """
        Load CSV file and preprocess text
        Expected columns: 'sms_message' and 'label' (spam=1, ham=0)
        """
        print("=" * 50)
        print("Step 1: Loading and preprocessing data")
        print("=" * 50)
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Check columns
        if 'sms_message' not in df.columns:
            raise ValueError("CSV must have 'sms_message' column")
        if 'label' not in df.columns:
            raise ValueError("CSV must have 'label' column")
        
        # Clean texts
        print("Cleaning text messages...")
        df['cleaned_text'] = df['sms_message'].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Get texts and labels
        texts = df['cleaned_text'].tolist()
        labels = df['label'].tolist()
        
        # Split data — 80% train, 10% val, 10% test
        print("Splitting data into train/val/test sets (80/10/10)...")
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        # Build vocabulary from training data
        print("Building vocabulary...")
        self.build_vocabulary(train_texts)
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        print(f"Test samples: {len(test_texts)}")
        
        return {
            'train': (train_texts, train_labels),
            'val': (val_texts, val_labels),
            'test': (test_texts, test_labels)
        }
    
    def create_dataloaders(self, data_splits, batch_size=32):
        """
        Create PyTorch dataloaders for training
        """
        print("\n" + "=" * 50)
        print("Step 2: Creating dataloaders")
        print("=" * 50)
        
        dataloaders = {}
        
        for split_name, (texts, labels) in data_splits.items():
            dataset = SMSDataset(texts, labels, self.word2idx, self.max_length)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(split_name == 'train'),
                num_workers=0
            )
            dataloaders[split_name] = dataloader
            
        return dataloaders
    
    def save_vocabulary(self, path=None):
        """Save vocabulary for later use"""
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.pkl')
        _dir = os.path.dirname(os.path.abspath(path))
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'max_length': self.max_length
            }, f)
        print(f"Vocabulary saved to {path}")
    
    def load_vocabulary(self, path=None):
        """Load saved vocabulary"""
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.max_length = data['max_length']