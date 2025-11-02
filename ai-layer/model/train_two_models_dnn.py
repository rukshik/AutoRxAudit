"""
Two-Model Opioid Audit System - Deep Neural Network Training
=============================================================

Trains two separate deep neural networks:
1. Eligibility Model: Predicts clinical need for opioids (opioid_eligibility)
   - Uses 8 features EXCLUDING opioid prescription history
   
2. OUD Risk Model: Predicts Opioid Use Disorder risk (y_oud)
   - Uses 11 features INCLUDING opioid exposure patterns
   - Uses class weights to handle imbalance

Usage:
    python train_two_models_dnn.py --data-dir ../processed_data/1000
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
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
import warnings

warnings.filterwarnings("ignore")

# Set random seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ===========================
# Dataset Class
# ===========================
class OpioidDataset(Dataset):
    """Custom Dataset for opioid prediction"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ===========================
# Deep Neural Network
# ===========================
class DeepNet(nn.Module):
    """
    Deep Neural Network for binary classification
    
    Architecture:
        - Input layer (variable size)
        - Hidden layers with BatchNorm + ReLU + Dropout
        - Output layer (no activation, use BCEWithLogitsLoss)
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.3):
        super(DeepNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no sigmoid - will use BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))
        
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
        
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
            
            # Apply sigmoid to get probabilities and clamp to avoid NaN
            outputs_clamped = torch.clamp(outputs, min=-10, max=10)  # Prevent extreme values
            probs = torch.sigmoid(outputs_clamped).cpu().numpy()
            probs = np.clip(probs, 1e-7, 1-1e-7)  # Ensure valid probability range
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Handle edge cases for AUC calculation
    try:
        # Check for NaN values
        all_probs_array = np.array(all_probs)
        if np.any(np.isnan(all_probs_array)) or np.any(np.isinf(all_probs_array)):
            print("  Warning: NaN or Inf detected in predictions, replacing with 0.5")
            all_probs_array = np.nan_to_num(all_probs_array, nan=0.5, posinf=0.999, neginf=0.001)
            all_probs = all_probs_array.tolist()
        
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5
    except Exception as e:
        print(f"  Warning: AUC calculation failed: {e}, using 0.5")
        auc = 0.5
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, auc, accuracy


def train_model_with_early_stopping(model, train_loader, val_loader, criterion, 
                                     optimizer, device, num_epochs=100, patience=15):
    """Train with early stopping"""
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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            print(f"  Best validation AUC: {best_val_auc:.4f}")
            break
    
    model.load_state_dict(best_model_state)
    return model, best_val_auc


def evaluate_test_set(model, test_loader, device):
    """Evaluate on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze()
            
            # Apply sigmoid to get probabilities and clamp to avoid NaN
            outputs_clamped = torch.clamp(outputs, min=-10, max=10)  # Prevent extreme values
            probs = torch.sigmoid(outputs_clamped).cpu().numpy()
            probs = np.clip(probs, 1e-7, 1-1e-7)  # Ensure valid probability range
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays and handle NaN values
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Replace any NaN values in probabilities
    all_probs = np.nan_to_num(all_probs, nan=0.5)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
    }
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return metrics, cm, all_preds, all_probs, all_labels


def train_single_model(train_df, val_df, test_df, target, feature_cols, 
                       model_name, output_dir, device):
    """Train a single DNN model"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING DNN: {model_name}")
    print(f"Target: {target}")
    print(f"{'='*80}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    train_dist = train_df[target].value_counts()
    val_dist = val_df[target].value_counts()
    test_dist = test_df[target].value_counts()
    
    print(f"  Training:   {train_dist.to_dict()} "
          f"({train_df[target].value_counts(normalize=True).round(3).to_dict()})")
    print(f"  Validation: {val_dist.to_dict()} "
          f"({val_df[target].value_counts(normalize=True).round(3).to_dict()})")
    print(f"  Test:       {test_dist.to_dict()} "
          f"({test_df[target].value_counts(normalize=True).round(3).to_dict()})")
    
    # Check for sufficient data
    if train_dist.min() < 2:
        print(f"\n⚠ Warning: Insufficient minority class samples")
        return None, None
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    
    # Prepare data - encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    # Create copies to avoid modifying original dataframes
    train_df_encoded = train_df[feature_cols].copy()
    val_df_encoded = val_df[feature_cols].copy()
    test_df_encoded = test_df[feature_cols].copy()
    
    # Encode categorical columns (like 'insurance')
    label_encoders = {}
    for col in feature_cols:
        if train_df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            train_df_encoded[col] = le.fit_transform(train_df_encoded[col])
            val_df_encoded[col] = le.transform(val_df_encoded[col])
            test_df_encoded[col] = le.transform(test_df_encoded[col])
            label_encoders[col] = le
    
    X_train = train_df_encoded.values
    y_train = train_df[target].values
    X_val = val_df_encoded.values
    y_val = val_df[target].values
    X_test = test_df_encoded.values
    y_test = test_df[target].values
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets and loaders
    train_dataset = OpioidDataset(X_train_scaled, y_train)
    val_dataset = OpioidDataset(X_val_scaled, y_val)
    test_dataset = OpioidDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train.astype(int))
    total = len(y_train)
    weight_for_0 = total / (2 * class_counts[0])
    weight_for_1 = total / (2 * class_counts[1])
    
    print(f"\nClass weights: [{weight_for_0:.3f}, {weight_for_1:.3f}]")
    
    # Create model
    input_dim = len(feature_cols)
    model = DeepNet(input_dim=input_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.3)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input: {input_dim} features")
    print(f"  Hidden layers: [128, 64, 32, 16]")
    print(f"  Output: 1 (binary classification)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Cap pos_weight to prevent numerical instability
    raw_pos_weight = weight_for_1 / weight_for_0
    pos_weight = torch.tensor([min(raw_pos_weight, 10.0)]).to(device)  # Cap at 10x
    print(f"  Pos weight: {raw_pos_weight:.3f} (capped at {pos_weight.item():.3f})")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # More stable than BCELoss
    
    # Use lower learning rate for stability with extreme class imbalance
    lr = 0.0001 if raw_pos_weight > 5.0 else 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"  Learning rate: {lr}")
    
    # Train
    trained_model, best_val_auc = train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=100, patience=15
    )
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")
    
    metrics, cm, preds, probs, labels = evaluate_test_set(trained_model, test_loader, device)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
    print(f"  FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")
    
    # Save model
    model_path = os.path.join(output_dir, f"dnn_{model_name}.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'input_dim': input_dim,
        'best_val_auc': best_val_auc,
    }, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        target: labels,
        'prediction_label': preds,
        'prediction_score': probs
    })
    pred_path = os.path.join(output_dir, f"dnn_{model_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved: {pred_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"dnn_{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    return trained_model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train two-model opioid audit system using Deep Neural Networks"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../processed_data/1000',
        help='Directory containing processed data (default: ../processed_data/1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save models and results (default: ./results)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_path = os.path.join(args.data_dir, "train_data.csv")
    val_path = os.path.join(args.data_dir, "validation_data.csv")
    test_path = os.path.join(args.data_dir, "test_data.csv")
    metadata_path = os.path.join(args.data_dir, "metadata.json")
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"  Training set: {train_df.shape}")
    print(f"  Validation set: {val_df.shape}")
    print(f"  Test set: {test_df.shape}")
    
    print("\n" + "="*80)
    print("TWO-MODEL OPIOID AUDIT SYSTEM - DNN TRAINING")
    print("="*80)
    print(f"\nAudit Logic: Flag if (Eligibility=NO) OR (OUD_Risk=HIGH)")
    
    # Model 1: Eligibility Model
    print("\n\n" + "="*80)
    print("MODEL 1: ELIGIBILITY MODEL (DNN)")
    print("="*80)
    
    eligibility_features = metadata['eligibility_features']
    eligibility_model, eligibility_metrics = train_single_model(
        train_df, val_df, test_df,
        target='opioid_eligibility',
        feature_cols=eligibility_features,
        model_name='eligibility_model',
        output_dir=args.output_dir,
        device=device
    )
    
    # Model 2: OUD Risk Model
    print("\n\n" + "="*80)
    print("MODEL 2: OUD RISK MODEL (DNN)")
    print("="*80)
    
    oud_features = metadata['oud_features']
    oud_model, oud_metrics = train_single_model(
        train_df, val_df, test_df,
        target='y_oud',
        feature_cols=oud_features,
        model_name='oud_risk_model',
        output_dir=args.output_dir,
        device=device
    )
    
    # Summary
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    if eligibility_metrics:
        print("\n1. Eligibility Model (DNN):")
        print(f"   AUC: {eligibility_metrics['auc']:.4f}")
        print(f"   F1:  {eligibility_metrics['f1']:.4f}")
    
    if oud_metrics:
        print("\n2. OUD Risk Model (DNN):")
        print(f"   AUC: {oud_metrics['auc']:.4f}")
        print(f"   F1:  {oud_metrics['f1']:.4f}")
    
    print(f"\n✓ All results saved to: {args.output_dir}")
    print("\nNext step: Compare all models")
    print("  python compare_all_models.py --results-dir ./results")


if __name__ == "__main__":
    main()
