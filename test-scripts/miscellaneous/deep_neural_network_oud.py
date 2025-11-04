"""
Deep Neural Network for OUD Prediction using PyTorch

This script trains a deep neural network on SHAP-selected features to predict:
  - y_oud: Opioid Use Disorder
  - will_get_opioid_rx: Opioid Prescription

Uses proper ML workflow:
  1. Train on training set (56.7K records)
  2. Validate on validation set (6.3K records) for early stopping
  3. Final evaluation on test set (30K records)

Results can be compared with PyCaret traditional ML models.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===========================
# Custom Dataset Class
# ===========================
class OUDDataset(Dataset):
    """Custom Dataset for OUD prediction"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ===========================
# Deep Neural Network Model
# ===========================
class DeepOUDNet(nn.Module):
    """
    Deep Neural Network for OUD Prediction
    
    Architecture:
        - Input layer (14 features)
        - Hidden layer 1: 128 neurons + BatchNorm + ReLU + Dropout
        - Hidden layer 2: 64 neurons + BatchNorm + ReLU + Dropout
        - Hidden layer 3: 32 neurons + BatchNorm + ReLU + Dropout
        - Hidden layer 4: 16 neurons + BatchNorm + ReLU + Dropout
        - Output layer: 1 neuron + Sigmoid (binary classification)
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.3):
        super(DeepOUDNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ===========================
# Training Functions
# ===========================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Store predictions
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, auc, accuracy, all_labels, all_preds, all_probs


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=100, patience=15):
    """
    Train the model with early stopping
    
    Args:
        patience: Number of epochs to wait before early stopping
    """
    best_val_auc = 0
    best_model_state = None
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    print(f"\nTraining Deep Neural Network...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Device: {device}")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_auc, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered after {epoch+1} epochs")
            print(f"  Best validation AUC: {best_val_auc:.4f}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_aucs, best_val_auc


def evaluate_test_set(model, test_loader, device):
    """Comprehensive evaluation on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze()
            
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


# ===========================
# Main Training Pipeline
# ===========================
def main():
    """Main training and evaluation pipeline"""
    
    # Define paths
    DATA_DIR = os.path.join("..", "processed_data")
    TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
    VAL_PATH = os.path.join(DATA_DIR, "validation_data.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test_data.csv")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"  Training set: {train_df.shape}")
    print(f"  Validation set: {val_df.shape}")
    print(f"  Test set: {test_df.shape}")
    
    targets = ["y_oud", "will_get_opioid_rx"]
    
    for target in targets:
        print(f"\n{'='*80}")
        print(f"DEEP NEURAL NETWORK - TARGET: {target}")
        print(f"{'='*80}")
        
        # Check class balance
        print(f"\nClass distribution:")
        print(f"  Training:   {train_df[target].value_counts().to_dict()}")
        print(f"  Validation: {val_df[target].value_counts().to_dict()}")
        print(f"  Test:       {test_df[target].value_counts().to_dict()}")
        
        # Prepare features and labels
        feature_cols = [col for col in train_df.columns 
                       if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]]
        
        print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df[target].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df[target].values
        
        # Standardize features
        print("\nStandardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Create datasets and dataloaders
        train_dataset = OUDDataset(X_train_scaled, y_train)
        val_dataset = OUDDataset(X_val_scaled, y_val)
        test_dataset = OUDDataset(X_test_scaled, y_test)
        
        batch_size = 256
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = len(feature_cols)
        model = DeepOUDNet(input_dim=input_dim, 
                          hidden_dims=[128, 64, 32, 16],
                          dropout_rate=0.3).to(device)
        
        print(f"\nModel architecture:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Train model
        trained_model, train_losses, val_losses, val_aucs, best_val_auc = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=100, patience=15
        )
        
        # Evaluate on test set
        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION ON TEST SET ({len(y_test)} samples)")
        print(f"{'='*80}")
        
        test_results = evaluate_test_set(trained_model, test_loader, device)
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1 Score:  {test_results['f1']:.4f}")
        print(f"  ROC AUC:   {test_results['auc']:.4f}")
        
        cm = test_results['confusion_matrix']
        print(f"\nConfusion Matrix (Test Set):")
        print(f"  TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
        print(f"  FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")
        
        # Save model
        model_path = f"dnn_model_{target}.pth"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'scaler': scaler,
            'feature_cols': feature_cols,
            'architecture': {
                'input_dim': input_dim,
                'hidden_dims': [128, 64, 32, 16],
                'dropout_rate': 0.3
            },
            'test_metrics': test_results
        }, model_path)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save test predictions
        predictions_df = pd.DataFrame({
            'subject_id': test_df['subject_id'],
            target: test_results['labels'],
            'prediction_label': test_results['predictions'],
            'prediction_score': test_results['probabilities']
        })
        pred_file = f"dnn_test_predictions_{target}.csv"
        predictions_df.to_csv(pred_file, index=False)
        print(f"✓ Test predictions saved to: {pred_file}")
        
        print(f"\n{'='*80}")
        print(f"Completed training for {target}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
