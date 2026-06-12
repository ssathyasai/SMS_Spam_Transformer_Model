"""
Configuration file for the SMS Spam Detector
Contains all hyperparameters and model settings
"""

class Config:
    # Model Architecture
    ENCODER_LAYERS = 3
    DECODER_LAYERS = 3
    MODEL_SIZE = 256  # d_model
    FEEDFORWARD_SIZE = 512
    ATTENTION_HEADS = 8
    MEMORY_LENGTH = 10
    DROPOUT_RATE = 0.25
    LINEAR_DROPOUT = 0.3
    
    # Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    WARMUP_STEPS = 20
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 15
    
    # Data Parameters
    MAX_SEQUENCE_LENGTH = 150
    EMBEDDING_DIM = 300  # GloVe 300d
    VOCAB_SIZE = 50000
    
    # Paths
    MODEL_SAVE_PATH = 'backend/models/spam_transformer.pth'
    GLOVE_PATH = 'backend/glove.840B.300d.txt'  # Will download if not exists
    
    # Device
    DEVICE = 'cuda'  # Will be auto-detected
    
    @classmethod
    def get_device(cls):
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')