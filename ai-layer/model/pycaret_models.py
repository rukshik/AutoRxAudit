"""
AutoRxAudit - PyCaret for classical ML models
=================================================

Trains two separate models:
1. Eligibility Model: Predicts clinical need for opioids (opioid_eligibility)

2. OUD Risk Model: Predicts Opioid Use Disorder risk (y_oud)

Usage:
    python pycaret_models.py --data-dir ../processed_data/1000
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from pycaret.classification import (
    setup,
    compare_models,
    tune_model,
    predict_model,
    save_model,
)
import warnings

warnings.filterwarnings("ignore")

# load data
def load_data(data_dir):
    train_path = os.path.join(data_dir, "train_data.csv")
    val_path = os.path.join(data_dir, "validation_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return train_df, val_df, test_df, metadata


# train a model
def train_model(train_df, val_df, test_df, target, feature_cols, model_name, output_dir):

    # Show features being used
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    
    # Prepare data (keep only features and target)
    train_modeling = train_df[feature_cols + [target]].copy().reset_index(drop=True)
    val_modeling = val_df[feature_cols + [target]].copy().reset_index(drop=True)
    test_modeling = test_df[feature_cols + [target]].copy().reset_index(drop=True)
    
    # Combine train and validation for PyCaret setup
    train_val_combined = pd.concat([train_modeling, val_modeling], ignore_index=True)
    
    # Setup PyCaret
    clf = setup(
        data=train_val_combined,
        target=target,
        test_data=test_modeling,
        train_size=len(train_modeling) / len(train_val_combined),
        preprocess=True,
        session_id=42,
        verbose=False,
        index=False, 
    )
    
    # Compare models
    models_to_test = ["lr", "rf", "gbc", "ada", "et", "lightgbm", "xgboost"]

    best_models = compare_models(
        include=models_to_test,
        sort="AUC",
        n_select=3,
    )
    
    # Get best model
    best_model = best_models[0] if isinstance(best_models, list) else best_models
    print(f"Best model: {best_model}")
    
    # Tune hyperparameters
    print(f"Tuning hyperparameters...")
    print(f"Optimization metric: AUC")
    print(f"Iterations: 30")
    
    tuned_model = tune_model(best_model, optimize="AUC", n_iter=30)
 
    # Use raw_score=True to get probability for each class
    test_predictions = predict_model(tuned_model, data=test_modeling, raw_score=True)
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    
    y_true = test_predictions[target]
    y_pred = test_predictions['prediction_label']
    # Use prediction_score_1 for probability of positive class (class 1)
    y_proba = test_predictions['prediction_score_1']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Test Set Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['auc']:.4f}")
    
    print(f"Confusion Matrix:")
    print(f"TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
    print(f"FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")
    
    # Save model
    model_path = os.path.join(output_dir, f"pycaret_{model_name}")
    save_model(tuned_model, model_path)
    
    # Save predictions
    pred_path = os.path.join(output_dir, f"pycaret_{model_name}_predictions.csv")
    test_predictions.to_csv(pred_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"pycaret_{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return tuned_model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_df, val_df, test_df, metadata = load_data(args.data_dir)
    
   
    
    # Model 1: Eligibility Model   
    eligibility_features = metadata['eligibility_features']
    eligibility_model, eligibility_metrics = train_model(
        train_df, val_df, test_df,
        target='opioid_eligibility',
        feature_cols=eligibility_features,
        model_name='eligibility_model',
        output_dir=args.output_dir
    )
    
    # Model 2: OUD Risk Model
    oud_features = metadata['oud_features']
    oud_model, oud_metrics = train_model(
        train_df, val_df, test_df,
        target='y_oud',
        feature_cols=oud_features,
        model_name='oud_risk_model',
        output_dir=args.output_dir
    )
    
    # Metrics

    print("Eligibility Model (PyCaret):")
    print(f"AUC: {eligibility_metrics['auc']:.4f}")
    print(f"F1:  {eligibility_metrics['f1']:.4f}")
    print(f"Features: {len(eligibility_features)}")
    
    print("OUD Risk Model (PyCaret):")
    print(f"AUC: {oud_metrics['auc']:.4f}")
    print(f"F1:  {oud_metrics['f1']:.4f}")
    print(f"Features: {len(oud_features)}")
    

if __name__ == "__main__":
    main()
