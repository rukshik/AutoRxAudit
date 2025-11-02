
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve

def main():
    print("[DEBUG] Starting modeling script...")
    # Load features
    features_csv = "mimiciv_demo_oud_features.csv"
    try:
        df = pd.read_csv(features_csv)
        print(f"[DEBUG] Loaded {features_csv} with shape {df.shape}")
    except Exception as e:
        print(f"[ERROR] Could not read {features_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    categorical = ["gender", "race", "insurance"]
    numeric = [
        "age_at_first_admit",
        "n_hospital_admits", "avg_los_days", "total_los_days",
        "any_opioid_flag", "opioid_rx_count", "distinct_opioids",
        "opioid_exposure_days", "any_benzo_flag"
    ]

    targets = [
        ("Opioid prescription (will_get_opioid_rx)", "will_get_opioid_rx"),
        ("OUD diagnosis (y_oud)", "y_oud"),
    ]

    print("[DEBUG] First 5 rows of features:")
    print(df.head())
    for target_name, target_col in targets:
        print(f"[DEBUG] Unique values for {target_col}: {df[target_col].unique()}")
        print(f"\n==============================\nModeling for: {target_name}\n==============================")
        data = df.dropna(subset=[target_col]).copy()
        X = data[categorical + numeric]
        y = data[target_col].astype(int)

        class_counts = y.value_counts()
        if class_counts.min() < 2 or len(class_counts) < 2:
            print("[WARNING] Very few samples in one or more classes. Results may be unreliable.")
            print(class_counts)

        # Train/test split (no stratify, so it works even with tiny classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", StandardScaler(), numeric),
            ]
        )

        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]

        for name, model in models:
            print(f"\n=== {name} ===")
            clf = Pipeline(steps=[
                ("pre", pre),
                ("clf", model)
            ])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if hasattr(clf.named_steps['clf'], "predict_proba"):
                y_proba = clf.predict_proba(X_test)[:, 1]
            else:
                y_proba = clf.decision_function(X_test)

            print(classification_report(y_test, y_pred, digits=3))

            try:
                roc = roc_auc_score(y_test, y_proba)
                print(f"ROC-AUC: {roc:.3f}")
            except Exception as e:
                print(f"ROC-AUC not available: {e}")

            try:
                pr_auc = average_precision_score(y_test, y_proba)
                print(f"PR-AUC: {pr_auc:.3f}")
            except Exception as e:
                print(f"PR-AUC not available: {e}")

if __name__ == "__main__":
    print("[DEBUG] __main__ block reached. Calling main()...")
    sys.exit(main())
