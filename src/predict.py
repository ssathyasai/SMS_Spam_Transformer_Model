"""
Prediction module for SMS Spam Detection
Handles single and batch predictions
"""

import torch
import numpy as np
import pandas as pd
from preprocess import DataPreprocessor
import os

class SpamPredictor:
    """Handles spam predictions"""
    
    def __init__(self, model, preprocessor, device, config, threshold=0.5):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.config = config
        self.threshold = threshold
        self.model.eval()
    
    def predict_single(self, text):
        """
        Predict whether a single SMS is spam or ham
        
        Steps:
        1. Clean the text
        2. Tokenize and convert to indices
        3. Pad to max length
        4. Run through model
        5. Return prediction and confidence
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
        
        # Determine prediction using tuned threshold
        is_spam = probability > self.threshold
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