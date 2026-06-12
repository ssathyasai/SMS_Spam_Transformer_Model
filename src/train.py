"""
Training module for Spam Transformer
Handles model training, validation, and saving
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
from tqdm import tqdm

class Trainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Loss function with class weights
        self.criterion = nn.BCELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_threshold = 0.5
        
    def _get_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.WARMUP_STEPS:
                return float(step) / float(max(1, self.config.WARMUP_STEPS))
            return max(0.0, float(self.config.WARMUP_STEPS) / float(step))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Calculate loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = (output > 0.5).float()
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, tune_threshold=False):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                probabilities.extend(output.detach().cpu().numpy())
                true_labels.extend(target.cpu().numpy())
        
        # Choose threshold using validation set when requested.
        threshold = self.best_threshold
        if tune_threshold:
            best_f1 = -1.0
            best_t = 0.5
            for t in np.arange(0.1, 0.91, 0.05):
                preds_t = (np.array(probabilities) > t).astype(float)
                f1_t = f1_score(true_labels, preds_t, zero_division=0)
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_t = float(t)
            self.best_threshold = best_t
            threshold = best_t

        predictions = (np.array(probabilities) > threshold).astype(float)

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': threshold
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting Model Training")
        print("=" * 60)
        
        best_f1 = -1
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Train
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            train_time = time.time() - start_time
            
            # Validate
            val_metrics = self.validate(val_loader, tune_threshold=True)
            patience_limit = getattr(self.config, 'EARLY_STOPPING_PATIENCE', None)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
            print(f"Best threshold: {val_metrics['threshold']:.2f}")
            print(f"Time: {train_time:.2f}s")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_model()
                patience_counter = 0
                print(f"[OK] New best model saved! F1-Score: {best_f1:.4f}")
            else:
                if patience_limit is not None:
                    patience_counter += 1
                    print(f"F1-Score did not improve. Patience: {patience_counter}/{patience_limit}")
                else:
                    print("F1-Score did not improve.")
            
            # Early stopping (only if EARLY_STOPPING_PATIENCE is set)
            if patience_limit is not None and patience_counter >= patience_limit:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
        
        print("\n" + "=" * 60)
        print(f"Training Complete! Best F1-Score: {best_f1:.4f}")
        print("=" * 60)
        
        return best_f1
    
    def save_model(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_threshold': self.best_threshold
        }
        
        os.makedirs(os.path.dirname(self.config.MODEL_SAVE_PATH), exist_ok=True)
        torch.save(checkpoint, self.config.MODEL_SAVE_PATH)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")
    
    def load_model(self, path=None):
        """Load model checkpoint"""
        if path is None:
            path = self.config.MODEL_SAVE_PATH
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_threshold = float(checkpoint.get('best_threshold', 0.5))
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"No model found at {path}")
            return False