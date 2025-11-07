"""
AutoRxAudit - Deep Neural Network

Use PyTorch

Traning two models

1. Eligibility Model: Predicts clinical need for opioids (opioid_eligibility)
   
2. OUD Risk Model: Predicts Opioid Use Disorder risk (y_oud)

Usage:
    python dnn_models.py --data-dir ../processed_data/1000
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

# Opioid Dataset Class (features and labels)
class OpioidDataset(Dataset):
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Deep Neural Network
# Binary classification
# Input layer - depends on feature set
# Hidden layers with BatchNormalization, ReLU and Dropout
# Output, no activation. 
class DeepNet(nn.Module):

    
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
        
        # Output layer (BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Training one epocj
def train_epoch(model, train_loader, criterion, optimizer, device):
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


# Model validaton
def validate(model, val_loader, criterion, device):
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
    best_val_auc = 0
    best_model_state = None
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        # print stats for for every 10 epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f" Epoch {epoch+1:3d}/{num_epochs}: "
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
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best validation AUC: {best_val_auc:.4f}")
            break
    
    model.load_state_dict(best_model_state)
    return model, best_val_auc


# Evaluate with test dataset
def evaluate_test_set(model, test_loader, device):
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

# train the model - called for eligibility and oud
def train_single_model(train_df, val_df, test_df, target, feature_cols, 
                       model_name, output_dir, device):
    
    print(f"TRAINING DNN: {model_name}")

    # Prepare data - encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
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
    
    # Create model
    input_dim = len(feature_cols)
    model = DeepNet(input_dim=input_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.3)
    model = model.to(device)
 
    # Loss and optimizer
    raw_pos_weight = weight_for_1 / weight_for_0
    pos_weight = torch.tensor([min(raw_pos_weight, 10.0)]).to(device)  # Cap at 10x
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # More stable than BCELoss
    
    # Use lower learning rate for stability with extreme class imbalance (in OUD case as postive cases are too few)
    lr = 0.0001 if raw_pos_weight > 5.0 else 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train
    trained_model, best_val_auc = train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=100, patience=15
    )

    metrics, cm, preds, probs, labels = evaluate_test_set(trained_model, test_loader, device)
    
    print(f"Test Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['auc']:.4f}")
    
    print(f"Confusion Matrix:")
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
    
    # Save predictions
    pred_df = pd.DataFrame({
        target: labels,
        'prediction_label': preds,
        'prediction_score': probs
    })
    pred_path = os.path.join(output_dir, f"dnn_{model_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"dnn_{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return trained_model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_path = os.path.join(args.data_dir, "train_data.csv")
    val_path = os.path.join(args.data_dir, "validation_data.csv")
    test_path = os.path.join(args.data_dir, "test_data.csv")
    metadata_path = os.path.join(args.data_dir, "metadata.json")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Model 1: Eligibility Model
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
    oud_features = metadata['oud_features']
    oud_model, oud_metrics = train_single_model(
        train_df, val_df, test_df,
        target='y_oud',
        feature_cols=oud_features,
        model_name='oud_risk_model',
        output_dir=args.output_dir,
        device=device
    )
    
    # print metrics    
    if eligibility_metrics:
        print("Eligibility Model (DNN):")
        print(f"   AUC: {eligibility_metrics['auc']:.4f}")
        print(f"   F1:  {eligibility_metrics['f1']:.4f}")
    
    if oud_metrics:
        print("OUD Risk Model (DNN):")
        print(f"   AUC: {oud_metrics['auc']:.4f}")
        print(f"   F1:  {oud_metrics['f1']:.4f}")
    

if __name__ == "__main__":
    main()
