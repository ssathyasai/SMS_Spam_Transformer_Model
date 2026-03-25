#!/usr/bin/env python
"""
Main entry point for SMS Spam Detector
Handles training, testing, and launching the server
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time

# Resolve backend package so imports use backend/* not the repo-root model.py
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, 'backend')

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import torch
        import flask
        import pandas
        import numpy
        import sklearn
        print("[OK] All dependencies found")
        return True
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("\nPlease install requirements with:")
        print("pip install -r requirements.txt")
        return False

def create_sample_data():
    """Create sample dataset if not exists"""
    import pandas as pd
    
    # Sample SMS messages for training
    sample_data = {
        'sms_message': [
            "Congratulations! You've won a $1000 gift card. Click here to claim now!",
            "Hey, are we still meeting for lunch tomorrow?",
            "URGENT: Your account has been compromised. Verify now: http://fake.com",
            "Don't forget to buy milk on your way home",
            "FREE entry to the casino! Click here to get your bonus",
            "The meeting has been rescheduled to 3pm",
            "You have won a lottery prize of $5000. Send your details to claim",
            "Can you pick up the kids from school today?",
            "Claim your prize now! Limited time offer!",
            "What time does the movie start?",
            "You've been selected for a special offer! Click now!",
            "Let's catch up this weekend",
            "Congratulations! You are our lucky winner!",
            "I'll be there in 10 minutes",
            "WINNER! Claim your free iPhone now!",
            "The report is ready for review",
            "URGENT: Your subscription expires today. Renew now!",
            "Have you seen the new Marvel movie?",
            "Exclusive deal just for you! 90% off today only!",
            "Can you send me the presentation slides?"
        ],
        'label': [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/sms_spam.csv'
    df.to_csv(csv_path, index=False)
    print(f"Sample dataset created at {csv_path}")
    return csv_path

def download_glove():
    """Download GloVe embeddings if not exists"""
    glove_path = 'backend/glove.840B.300d.txt'
    if not os.path.exists(glove_path):
        print("Downloading GloVe embeddings (this may take a while)...")
        print("Note: This is a large file (~2GB). You can also use a smaller version.")
        
        # Download a smaller version for testing
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        import requests
        import zipfile
        
        try:
            # Download smaller GloVe version
            print("Downloading GloVe 6B (100d) as an alternative...")
            response = requests.get("https://nlp.stanford.edu/data/glove.6B.zip", stream=True)
            
            zip_path = "backend/glove.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('backend/')
            
            # Use 300d version if exists, otherwise use 100d
            if os.path.exists('backend/glove.6B.300d.txt'):
                os.rename('backend/glove.6B.300d.txt', glove_path)
            else:
                os.rename('backend/glove.6B.100d.txt', glove_path)
            
            os.remove(zip_path)
            print("[OK] GloVe embeddings downloaded successfully")
        except Exception as e:
            print(f"[!] Error downloading GloVe: {e}")
            print("You can download manually from https://nlp.stanford.edu/projects/glove/")
            return False
    return True

def train_model():
    """Train the spam detection model"""
    print("\n" + "=" * 60)
    print("Training Spam Detection Model")
    print("=" * 60)
    
    # Create sample data if needed
    data_path = 'data/sms_spam.csv'
    if not os.path.exists(data_path):
        print("Creating sample dataset...")
        data_path = create_sample_data()
    
    # Import training modules (backend must shadow repo-root model.py)
    if _BACKEND_DIR not in sys.path:
        sys.path.insert(0, _BACKEND_DIR)
    from config import Config
    from model import SpamTransformerWithEmbeddings
    from preprocess import DataPreprocessor
    from train import Trainer
    
    import torch
    
    # Setup
    device = Config.get_device()
    print(f"Using device: {device}")
    
    # Preprocess data
    preprocessor = DataPreprocessor(
        max_length=Config.MAX_SEQUENCE_LENGTH,
        vocab_size=Config.VOCAB_SIZE
    )
    
    data_splits = preprocessor.load_and_preprocess_data(data_path)
    dataloaders = preprocessor.create_dataloaders(data_splits, Config.BATCH_SIZE)
    
    # Save vocabulary
    preprocessor.save_vocabulary()
    
    # Initialize model
    model = SpamTransformerWithEmbeddings(
        vocab_size=len(preprocessor.word2idx),
        d_model=Config.MODEL_SIZE,
        num_heads=Config.ATTENTION_HEADS,
        num_encoder_layers=Config.ENCODER_LAYERS,
        num_decoder_layers=Config.DECODER_LAYERS,
        d_ff=Config.FEEDFORWARD_SIZE,
        dropout=Config.DROPOUT_RATE,
        memory_length=Config.MEMORY_LENGTH
    ).to(device)
    
    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Train model
    trainer = Trainer(model, device, Config)
    best_f1 = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        Config.NUM_EPOCHS
    )
    # Ensure evaluation uses best saved checkpoint (not last epoch)
    trainer.load_model()
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    test_metrics = trainer.validate(dataloaders['test'])
    print(f"Test Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  Threshold: {test_metrics['threshold']:.2f}")
    
    print("\n[OK] Model training complete!")
    return True

def launch_server():
    """Launch the Flask server and open browser"""
    print("\n" + "=" * 60)
    print("Launching SMS Spam Detector")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'backend/models/spam_transformer.pth'
    if not os.path.exists(model_path):
        print("Model not found. Training model first...")
        if not train_model():
            print("Failed to train model. Exiting.")
            return False
    
    # Start server in subprocess
    server_process = subprocess.Popen(
        [sys.executable, 'backend/app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("Starting server...")
    time.sleep(3)
    
    # Open browser
    url = "http://localhost:5000"
    print(f"\nOpening browser at {url}")
    webbrowser.open(url)
    
    print("\nPress Ctrl+C to stop the server...")
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server_process.terminate()
        server_process.wait()
        print("Server stopped.")

def main():
    parser = argparse.ArgumentParser(description='SMS Spam Detector')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--serve', action='store_true', help='Launch the web server')
    parser.add_argument('--check', action='store_true', help='Check requirements')
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        return
    
    if args.check:
        return
    
    if args.train:
        train_model()
    elif args.serve:
        launch_server()
    else:
        # Default: launch server
        launch_server()

if __name__ == '__main__':
    main()