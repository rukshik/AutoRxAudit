import pandas as pd
import numpy as np
from pycaret.classification import *
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# Main training / evaluation pipeline
# ------------------------------


def main():
    """
    Load SHAP-selected features from CSV and train/evaluate multiple classifiers using PyCaret on two targets:
      - y_oud: Opioid Use Disorder label (from ICD codes)
      - will_get_opioid_rx: whether the patient will ever receive an opioid prescription

    This version uses features selected by SHAP analysis instead of manual feature selection.

    PyCaret will automatically:
      - Test multiple models on SHAP-selected features
      - Handle preprocessing (scaling, encoding, etc.)
      - Perform hyperparameter tuning
      - Provide comprehensive evaluation metrics
      - Generate model comparison plots
    """
    # Load SHAP-selected data
    df = pd.read_csv("mimiciv_demo_oud_features_with_shap_selection.csv")

    targets = ["y_oud", "will_get_opioid_rx"]

    for target in targets:
        print(f"\n{'='*60}")
        print(f"MODELING TARGET: {target}")
        print(f"{'='*60}")

        # Quick look at class balance
        print(f"Target distribution: {np.bincount(df[target])}")
        print(
            f"Class balance: {df[target].value_counts(normalize=True).round(3).to_dict()}"
        )

        # Show SHAP-selected features
        feature_cols = [
            col
            for col in df.columns
            if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]
        ]
        print(f"\nSHAP-selected features ({len(feature_cols)}): {feature_cols}")

        # Prepare data for PyCaret (drop subject_id, keep target)
        modeling_df = df.drop(columns=["subject_id"], errors="ignore")

        # Setup PyCaret environment
        print(f"\nSetting up PyCaret environment for {target}...")

        # Check if target is too imbalanced for modeling
        target_counts = df[target].value_counts()
        min_class_count = target_counts.min()

        if min_class_count < 2:
            print(
                f"Warning: Target '{target}' has only {min_class_count} samples in minority class."
            )
            print("Skipping this target due to insufficient data for modeling.")
            continue

        clf = setup(
            data=modeling_df,
            target=target,
            train_size=0.67,  # 67% train, 33% test (same as original)
            test_data=None,
            preprocess=True,  # Auto preprocessing
            session_id=42,  # For reproducibility
        )

        # Compare available models (using only core models)
        print(f"\nComparing available models for {target}...")
        best_models = compare_models(
            include=[
                "lr",  # Logistic Regression
                "rf",  # Random Forest
                "gbc",  # Gradient Boosting
                "ada",  # AdaBoost
                "et",  # Extra Trees
                "knn",  # K-Nearest Neighbors
                "nb",  # Naive Bayes
                "qda",  # Quadratic Discriminant Analysis
                "lda",  # Linear Discriminant Analysis
                "svm",  # Support Vector Machine
                "dt",  # Decision Tree
            ],
            sort="AUC",  # Sort by AUC score
            n_select=5,  # Select top 5 models
        )

        # Get the best model
        best_model = best_models[0] if isinstance(best_models, list) else best_models
        print(f"\nBest model for {target}: {best_model}")

        # Tune the best model
        print(f"\nTuning hyperparameters for best model...")
        tuned_model = tune_model(best_model, optimize="AUC", n_iter=20)

        # Evaluate the tuned model
        print(f"\nEvaluating tuned model...")
        evaluate_model(tuned_model)

        # Generate predictions and final metrics
        print(f"\nFinal model performance:")
        predictions = predict_model(tuned_model)

        # Save the model
        model_name = f"best_model_shap_{target.replace('_', '_')}"
        save_model(tuned_model, model_name)
        print(f"Model saved as: {model_name}")

        print(f"\nCompleted modeling for {target}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
