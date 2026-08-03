"""
Prediction module for SMS Spam Detection
Handles single and batch predictions
"""

import torch
import numpy as np
import pandas as pd
import os
import re
from preprocess import DataPreprocessor

class SpamPredictor:
    """Handles spam predictions with calibrated threshold and heuristic signal booster"""
    
    def __init__(self, model, preprocessor, device, config, threshold=0.5):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.config = config
        # Cap threshold at 0.50 for standard calibrated decision boundary
        self.threshold = 0.50 if threshold > 0.55 else threshold
        self.model.eval()
    
    def _detect_tricky_spam_boost(self, text):
        """Detect tricky/obfuscated spam signals (phishing links, leetspeak, urgent account alerts)"""
        tricky_patterns = [
            r'c[1!i]ick', r'fr[3e][3e]', r'w[1!i]n', r'c[4a]sh', r'p[r1!]ze',
            r'http\S+', r'bit\.ly', r'\.info', r'\.xyz', r'on\s+hold',
            r'account\s+locked', r'verify\s+your', r'subscription\s+failed',
            r'bank\s+account', r'claim\s+your', r'lotto', r'lottery'
        ]
        text_lower = text.lower()
        signal_count = sum(1 for pat in tricky_patterns if re.search(pat, text_lower))
        return signal_count

    def _detect_legit_ham_signals(self, text):
        """Detect legitimate work, interview, meeting, personal, transactional, delivery, and utility ham patterns"""
        ham_patterns = [
            # Work & Career
            r'interview\s+(for|at|on|scheduled)', r'intern(ship)?\b', r'candidate', r'recruitment', r'hr\s+team',
            r'teams\s+link', r'zoom\s+link', r'meet\s+link', r'meeting\s+id', r'scheduled\s+for',
            r'application\s+status', r'resume', r'verification\s+code', r'otp\s+is', r'job\s+offer',
            # Personal & Casual
            r'call\s+me\b', r'reach\s+home', r'driving', r'see\s+you', r'let\s+me\s+know', r'pick\s+up',
            # Transactional & Order / Delivery / Payment updates
            r'order\s+(has\s+been\s+)?(confirmed|placed|shipped|delivered)', r'food\s+order', r'out\s+for\s+delivery',
            r'will\s+be\s+delivered', r'delivery\s+(expected|agent|boy|driver)', r'package\s+has\s+been',
            r'payment\s+(of|received|successful|confirmed)', r'account\s+(debited|credited)',
            r'transaction\s+(successful|id|reference)', r'booking\s+confirmed', r'cab\s+is\s+on\s+the\s+way',
            r'ride\s+confirmed', r'pnr\b', r'seat\s+no'
        ]
        text_lower = text.lower()
        return sum(1 for pat in ham_patterns if re.search(pat, text_lower))

    def predict_single(self, text):
        """
        Predict whether a single SMS is spam or ham
        
        Steps:
        1. Clean the text
        2. Tokenize and convert to indices
        3. Pad to max length
        4. Run through model
        5. Apply signal booster/dampener for tricky messages
        6. Return prediction and confidence
        """
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Tokenize
        tokens = cleaned_text.split()
        
        # Convert to indices
        indices = []
        for token in tokens:
            idx = self.preprocessor.word2idx.get(token, self.preprocessor.word2idx['<UNK>'])
            indices.append(idx)
        
        # Pad
        padding_length = self.preprocessor.max_length - len(indices)
        if padding_length > 0:
            indices = indices + [self.preprocessor.word2idx['<PAD>']] * padding_length
        else:
            indices = indices[:self.preprocessor.max_length]
        
        # Convert to tensor
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = output.item()
        
        # Apply heuristic booster / dampener for tricky messages
        spam_signals = self._detect_tricky_spam_boost(text)
        ham_signals = self._detect_legit_ham_signals(text)
        
        if spam_signals >= 1 and probability >= 0.35:
            probability = min(0.99, probability + 0.20 * spam_signals)
        elif ham_signals >= 1 and spam_signals == 0:
            probability = max(0.01, probability - 0.25 * ham_signals)
        
        # Determine prediction using tuned threshold
        is_spam = probability >= self.threshold
        label = "SPAM" if is_spam else "HAM"
        confidence = probability if is_spam else 1 - probability
        
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'prediction': label,
            'is_spam': is_spam,
            'confidence': confidence,
            'probability': probability
        }
    
    def predict_file(self, file_path):
        """
        Process file containing multiple messages
        Returns DataFrame with predictions and regenerated file
        """
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            # Try different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        
        # Process each line
        results = []
        spam_count = 0
        ham_count = 0
        
        print(f"Processing {len(lines)} messages...")
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                result = self.predict_single(line)
                results.append(result)
                
                if result['is_spam']:
                    spam_count += 1
                else:
                    ham_count += 1
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add statistics
        stats = {
            'total_messages': len(results),
            'spam_count': spam_count,
            'ham_count': ham_count,
            'spam_percentage': (spam_count / len(results) * 100) if results else 0,
            'ham_percentage': (ham_count / len(results) * 100) if results else 0
        }
        
        return df, stats
    
    def regenerate_file(self, input_path, output_path):
        """
        Process input file and generate output file with predictions
        """
        df, stats = self.predict_file(input_path)
        
        # Create output content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SMS SPAM DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Messages: {stats['total_messages']}\n")
            f.write(f"SPAM Messages: {stats['spam_count']} ({stats['spam_percentage']:.2f}%)\n")
            f.write(f"HAM Messages: {stats['ham_count']} ({stats['ham_percentage']:.2f}%)\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 80 + "\n")
            
            for idx, row in df.iterrows():
                f.write(f"\nMessage #{idx + 1}:\n")
                f.write(f"Original: {row['text'][:100]}...\n")
                f.write(f"Prediction: {row['prediction']}\n")
                f.write(f"Confidence: {row['confidence']:.2%}\n")
                f.write("-" * 40 + "\n")
        
        return stats