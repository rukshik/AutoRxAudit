import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
import warnings

warnings.filterwarnings("ignore")


def main():
    df = pd.read_csv("mimiciv_demo_oud_features_with_atc_rx.csv")
    targets = ["y_oud", "will_get_opioid_rx"]
    drop_cols = ["subject_id"] + targets
    X = df.drop(columns=drop_cols, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    for target in targets:
        print(f"\n=== Modeling {target} ===")
        y = df[target]
        print(f"Target distribution: {np.bincount(y)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
            (
                "GradientBoosting",
                GradientBoostingClassifier(n_estimators=100, random_state=42),
            ),
        ]
        for name, model in models:
            print(f"\n-- {name} --")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred
            )
            print(classification_report(y_test, y_pred, digits=3))
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
                pr_auc = average_precision_score(y_test, y_prob)
                print(f"ROC-AUC: {roc_auc:.3f}  PR-AUC: {pr_auc:.3f}")
            except Exception as e:
                print(f"AUC error: {e}")


if __name__ == "__main__":
    main()
