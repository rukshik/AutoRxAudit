"""
Two-Model Opioid Audit System - PyCaret Training
=================================================

Trains two separate models:
1. Eligibility Model: Predicts clinical need for opioids (opioid_eligibility)
   - Uses 8 features EXCLUDING opioid prescription history
   - No data leakage from past opioid prescriptions
   
2. OUD Risk Model: Predicts Opioid Use Disorder risk (y_oud)
   - Uses 11 features INCLUDING opioid exposure patterns
   - Opioid history is legitimate predictor for OUD risk

Audit Logic: Flag prescription if (Eligibility=NO) OR (OUD_Risk=HIGH)

Usage:
    python train_two_models_pycaret.py --data-dir ../processed_data/1000
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from pycaret.classification import *
import warnings

warnings.filterwarnings("ignore")


def load_data(data_dir):
    """Load train, validation, and test datasets"""
    train_path = os.path.join(data_dir, "train_data.csv")
    val_path = os.path.join(data_dir, "validation_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"  Training set: {train_df.shape}")
    print(f"  Validation set: {val_df.shape}")
    print(f"  Test set: {test_df.shape}")
    
    return train_df, val_df, test_df, metadata


def train_model(train_df, val_df, test_df, target, feature_cols, model_name, output_dir):
    """
    Train a model using PyCaret
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        target: Target column name ('y_oud' or 'opioid_eligibility')
        feature_cols: List of feature columns to use
        model_name: Name for saving the model
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {model_name}")
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
    
    # Check for sufficient minority class samples
    min_train = train_dist.min()
    if min_train < 2:
        print(f"\n⚠ Warning: Only {min_train} samples in minority class (training)")
        print("Skipping this model due to insufficient data.")
        return None
    
    # Show features being used
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    
    # Prepare data (keep only features and target)
    train_modeling = train_df[feature_cols + [target]].copy().reset_index(drop=True)
    val_modeling = val_df[feature_cols + [target]].copy().reset_index(drop=True)
    test_modeling = test_df[feature_cols + [target]].copy().reset_index(drop=True)
    
    # Combine train and validation for PyCaret setup
    train_val_combined = pd.concat([train_modeling, val_modeling], ignore_index=True)
    
    print(f"\nSetting up PyCaret environment...")
    print(f"  Training + Validation: {train_val_combined.shape[0]} samples")
    print(f"  Test (final holdout): {test_modeling.shape[0]} samples")
    
    # Setup PyCaret
    clf = setup(
        data=train_val_combined,
        target=target,
        test_data=test_modeling,
        train_size=len(train_modeling) / len(train_val_combined),
        preprocess=True,
        session_id=42,
        verbose=False,
        index=False,  # Reset index to avoid duplicate index errors
    )
    
    # Compare models
    print(f"\nComparing models...")
    print("  Models: Logistic Regression, Random Forest, Gradient Boosting,")
    print("          AdaBoost, Extra Trees, LightGBM")
    
    # Try to include xgboost if available
    models_to_test = ["lr", "rf", "gbc", "ada", "et", "lightgbm"]
    try:
        import xgboost
        models_to_test.append("xgboost")
        print("          XGBoost (available)")
    except ImportError:
        print("          (XGBoost not available - skipping)")
    
    best_models = compare_models(
        include=models_to_test,
        sort="AUC",
        n_select=3,
    )
    
    # Get best model
    best_model = best_models[0] if isinstance(best_models, list) else best_models
    print(f"\n✓ Best model: {best_model}")
    
    # Tune hyperparameters
    print(f"\nTuning hyperparameters...")
    print(f"  Optimization metric: AUC")
    print(f"  Iterations: 30")
    
    tuned_model = tune_model(best_model, optimize="AUC", n_iter=30)
    
    # Skip interactive plots
    print(f"\nHyperparameter tuning complete (skipping interactive plots)")
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION ON TEST SET ({test_modeling.shape[0]} samples)")
    print(f"{'='*80}")
    
    test_predictions = predict_model(tuned_model, data=test_modeling)
    
    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    
    y_true = test_predictions[target]
    y_pred = test_predictions['prediction_label']
    y_proba = test_predictions['prediction_score']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
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
    model_path = os.path.join(output_dir, f"pycaret_{model_name}")
    save_model(tuned_model, model_path)
    print(f"\n✓ Model saved: {model_path}.pkl")
    
    # Save predictions
    pred_path = os.path.join(output_dir, f"pycaret_{model_name}_predictions.csv")
    test_predictions.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved: {pred_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"pycaret_{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    return tuned_model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train two-model opioid audit system using PyCaret"
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_df, val_df, test_df, metadata = load_data(args.data_dir)
    
    print("\n" + "="*80)
    print("TWO-MODEL OPIOID AUDIT SYSTEM - PYCARET TRAINING")
    print("="*80)
    print(f"\nAudit Logic: Flag if (Eligibility=NO) OR (OUD_Risk=HIGH)")
    print(f"\nMetadata:")
    print(f"  Total records: {metadata['total_records']}")
    print(f"  Selected features: {metadata['selected_features']}")
    print(f"  Train: {metadata['train_records']}, Val: {metadata['validation_records']}, Test: {metadata['test_records']}")
    
    # Model 1: Eligibility Model
    print("\n\n" + "="*80)
    print("MODEL 1: ELIGIBILITY MODEL")
    print("="*80)
    print("Predicts clinical need for opioids based on patient history")
    print("Features EXCLUDE opioid prescription history (no data leakage)")
    
    eligibility_features = metadata['eligibility_features']
    eligibility_model, eligibility_metrics = train_model(
        train_df, val_df, test_df,
        target='opioid_eligibility',
        feature_cols=eligibility_features,
        model_name='eligibility_model',
        output_dir=args.output_dir
    )
    
    # Model 2: OUD Risk Model
    print("\n\n" + "="*80)
    print("MODEL 2: OUD RISK MODEL")
    print("="*80)
    print("Predicts risk of Opioid Use Disorder")
    print("Features INCLUDE opioid exposure (legitimate predictor for OUD risk)")
    
    oud_features = metadata['oud_features']
    oud_model, oud_metrics = train_model(
        train_df, val_df, test_df,
        target='y_oud',
        feature_cols=oud_features,
        model_name='oud_risk_model',
        output_dir=args.output_dir
    )
    
    # Summary
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    print("\n1. Eligibility Model (PyCaret):")
    print(f"   AUC: {eligibility_metrics['auc']:.4f}")
    print(f"   F1:  {eligibility_metrics['f1']:.4f}")
    print(f"   Features: {len(eligibility_features)}")
    
    print("\n2. OUD Risk Model (PyCaret):")
    print(f"   AUC: {oud_metrics['auc']:.4f}")
    print(f"   F1:  {oud_metrics['f1']:.4f}")
    print(f"   Features: {len(oud_features)}")
    
    print(f"\n✓ All results saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Run DNN training: python train_two_models_dnn.py")
    print("  2. Compare all models: python compare_all_models.py")
    print("  3. Implement audit system: python audit_system.py")


if __name__ == "__main__":
    main()
