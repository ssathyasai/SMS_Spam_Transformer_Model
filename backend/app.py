"""
Flask API server for SMS Spam Detector
Handles HTTP requests and serves predictions
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import torch
import pandas as pd
import json

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model import SpamTransformerWithEmbeddings
from preprocess import DataPreprocessor
from predict import SpamPredictor

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Global variables for model and predictor
model = None
predictor = None
device = None

# Resolve absolute paths so the server works regardless of cwd
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))  # .../deepseek_SMS_SPAM/backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)               # .../deepseek_SMS_SPAM

def load_model():
    """Load trained model and preprocessor"""
    global model, predictor, device
    
    print("=" * 60)
    print("Loading Spam Detection Model...")
    print("=" * 60)
    
    # Setup device
    device = Config.get_device()
    print(f"Using device: {device}")
    
    # Load preprocessor
    preprocessor = DataPreprocessor(max_length=Config.MAX_SEQUENCE_LENGTH)
    vocab_path = os.path.join(PROJECT_ROOT, 'backend', 'vocab.pkl')
    
    if os.path.exists(vocab_path):
        preprocessor.load_vocabulary(vocab_path)
        print("Preprocessor loaded successfully")
    else:
        print("ERROR: Vocabulary file not found. Please train the model first.")
        return False
    
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
    
    # Load trained weights
    model_save_path = os.path.join(PROJECT_ROOT, 'backend', 'models', 'spam_transformer.pth')
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = float(checkpoint.get('best_threshold', 0.5))
        print(f"Model weights loaded (threshold={best_threshold:.2f})")
    else:
        print("ERROR: Model weights not found. Please train the model first.")
        return False
    
    # Create predictor
    predictor = SpamPredictor(model, preprocessor, device, Config, threshold=best_threshold)
    print("Predictor ready!")
    print("=" * 60)
    
    return True

@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict spam/ham for a single message"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get prediction
        result = predictor.predict_single(message)
        
        # Format response
        response = {
            'success': True,
            'prediction': result['prediction'],
            'is_spam': result['is_spam'],
            'confidence': result['confidence'],
            'probability': result['probability'],
            'original_text': result['text'],
            'cleaned_text': result['cleaned_text']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict-file', methods=['POST'])
def predict_file():
    """Process uploaded file and return results"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save uploaded file temporarily
        temp_path = os.path.join(PROJECT_ROOT, 'backend', 'temp_upload.txt')
        file.save(temp_path)
        
        # Process file
        df, stats = predictor.predict_file(temp_path)
        
        # Prepare response
        results = []
        for idx, row in df.iterrows():
            results.append({
                'index': idx + 1,
                'text': row['text'][:100] + ('...' if len(row['text']) > 100 else ''),
                'prediction': row['prediction'],
                'confidence': row['confidence'],
                'is_spam': row['is_spam']
            })
        
        response = {
            'success': True,
            'stats': stats,
            'results': results,
            'total': len(results)
        }
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/regenerate-file', methods=['POST'])
def regenerate_file():
    """Process file and generate output file with predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save uploaded file
        input_path = os.path.join(PROJECT_ROOT, 'backend', 'temp_input.txt')
        output_path = os.path.join(PROJECT_ROOT, 'backend', 'output_report.txt')
        file.save(input_path)
        
        # Generate report
        stats = predictor.regenerate_file(input_path, output_path)
        
        # Read generated file
        with open(output_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        response = {
            'success': True,
            'stats': stats,
            'report': report_content
        }
        
        # Clean up
        os.remove(input_path)
        os.remove(output_path)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error regenerating file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load model before starting server
    if load_model():
        print("\n" + "=" * 60)
        print("Server starting on http://localhost:5000")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nERROR: Model not loaded. Please train the model first with:")
        print("python backend/train.py")