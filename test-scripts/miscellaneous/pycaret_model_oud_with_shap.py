import pandas as pd
import numpy as np
import os
from pycaret.classification import *
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# Main training / evaluation pipeline
# ------------------------------


def main():
    """
    Train/evaluate multiple classifiers using PyCaret on two targets:
      - y_oud: Opioid Use Disorder label (from ICD codes)
      - will_get_opioid_rx: whether the patient will ever receive an opioid prescription

    Uses proper ML workflow:
      1. Train on training set (56.7K records)
      2. Validate/tune on validation set (6.3K records)
      3. Final evaluation on test set (30K records)

    Features are selected by SHAP analysis (14 features from original 21).

    PyCaret will automatically:
      - Test multiple models on SHAP-selected features
      - Handle preprocessing (scaling, encoding, etc.)
      - Perform hyperparameter tuning using validation data
      - Provide comprehensive evaluation metrics
      - Generate model comparison plots
    """
    # Define paths to processed data
    DATA_DIR = os.path.join("..", "processed_data")
    TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
    VAL_PATH = os.path.join(DATA_DIR, "validation_data.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test_data.csv")

    # Load SHAP-selected data with proper train/val/test split
    print("Loading processed data with SHAP-selected features...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")

    targets = ["y_oud", "will_get_opioid_rx"]

    for target in targets:
        print(f"\n{'='*80}")
        print(f"MODELING TARGET: {target}")
        print(f"{'='*80}")

        # Quick look at class balance
        print(f"\nClass distribution:")
        print(f"  Training:   {train_df[target].value_counts().to_dict()} "
              f"({train_df[target].value_counts(normalize=True).round(3).to_dict()})")
        print(f"  Validation: {val_df[target].value_counts().to_dict()} "
              f"({val_df[target].value_counts(normalize=True).round(3).to_dict()})")
        print(f"  Test:       {test_df[target].value_counts().to_dict()} "
              f"({test_df[target].value_counts(normalize=True).round(3).to_dict()})")

        # Show SHAP-selected features
        feature_cols = [
            col
            for col in train_df.columns
            if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]
        ]
        print(f"\nSHAP-selected features ({len(feature_cols)}): {feature_cols}")

        # Check if target is too imbalanced for modeling
        target_counts = train_df[target].value_counts()
        min_class_count = target_counts.min()

        if min_class_count < 2:
            print(
                f"Warning: Target '{target}' has only {min_class_count} samples in minority class."
            )
            print("Skipping this target due to insufficient data for modeling.")
            continue

        # Prepare data for PyCaret (drop subject_id, keep features and target)
        train_modeling = train_df.drop(columns=["subject_id"], errors="ignore")
        val_modeling = val_df.drop(columns=["subject_id"], errors="ignore")
        test_modeling = test_df.drop(columns=["subject_id"], errors="ignore")

        # Combine train and validation for PyCaret setup, keep test completely separate
        train_val_combined = pd.concat([train_modeling, val_modeling], ignore_index=True)

        # Setup PyCaret environment
        print(f"\nSetting up PyCaret environment for {target}...")
        print(f"  Training + Validation: {train_val_combined.shape[0]} samples")
        print(f"  Test (final holdout): {test_modeling.shape[0]} samples")

        clf = setup(
            data=train_val_combined,
            target=target,
            test_data=test_modeling,  # Use test set as final holdout
            train_size=len(train_modeling) / len(train_val_combined),  # Maintain original train ratio
            preprocess=True,  # Auto preprocessing
            session_id=42,  # For reproducibility
            verbose=False,  # Reduce output noise
        )

        # Compare available models (using only core models)
        print(f"\nComparing available models for {target}...")
        print("  Models to test: Logistic Regression, Random Forest, Gradient Boosting,")
        print("                  AdaBoost, Extra Trees, LightGBM, XGBoost")
        
        best_models = compare_models(
            include=[
                "lr",      # Logistic Regression
                "rf",      # Random Forest
                "gbc",     # Gradient Boosting
                "ada",     # AdaBoost
                "et",      # Extra Trees
                "lightgbm", # LightGBM
                "xgboost", # XGBoost
            ],
            sort="AUC",  # Sort by AUC score
            n_select=3,  # Select top 3 models
        )

        # Get the best model
        best_model = best_models[0] if isinstance(best_models, list) else best_models
        print(f"\n✓ Best model for {target}: {best_model}")

        # Tune the best model using cross-validation on training data
        print(f"\nTuning hyperparameters for best model...")
        print(f"  Optimization metric: AUC")
        print(f"  Tuning iterations: 30")
        tuned_model = tune_model(best_model, optimize="AUC", n_iter=30)

        # Evaluate the tuned model on validation data (internal to PyCaret)
        print(f"\nEvaluating tuned model on validation data...")
        evaluate_model(tuned_model)

        # Final evaluation on completely held-out test set
        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION ON TEST SET ({test_modeling.shape[0]} samples)")
        print(f"{'='*80}")
        
        # Predict on test set
        test_predictions = predict_model(tuned_model, data=test_modeling)
        
        # Calculate and display test set metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        y_true = test_predictions[target]
        y_pred = test_predictions['prediction_label']
        y_proba = test_predictions['prediction_score']
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"  ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix (Test Set):")
        print(f"  TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
        print(f"  FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")

        # Save the model
        model_name = f"best_model_{target}"
        save_model(tuned_model, model_name)
        print(f"\n✓ Model saved as: {model_name}.pkl")

        # Save test predictions
        test_predictions_file = f"test_predictions_{target}.csv"
        test_predictions.to_csv(test_predictions_file, index=False)
        print(f"✓ Test predictions saved to: {test_predictions_file}")

        print(f"\n{'='*80}")
        print(f"Completed modeling for {target}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
