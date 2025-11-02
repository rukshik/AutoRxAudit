"""
Data Splitting Script for Neural Network Training
Demonstrates proper data splitting strategy without SHAP dependency
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


def read_csv(rel_path: str, demo_dir: str) -> pd.DataFrame:
    path = os.path.join(demo_dir, rel_path)
    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def mode_or_nan(series: pd.Series):
    if series.empty:
        return np.nan
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.idxmax()


def compute_age_at_first_admit(
    patients: pd.DataFrame, admissions: pd.DataFrame
) -> pd.DataFrame:
    first_admit = (
        admissions.sort_values("admittime")
        .groupby("subject_id", as_index=False)
        .first()[["subject_id", "hadm_id", "admittime"]]
    )
    demo = patients.merge(first_admit, on="subject_id", how="left")
    demo["admittime"] = pd.to_datetime(demo["admittime"], errors="coerce")
    admit_year = demo["admittime"].dt.year
    demo["age_at_first_admit"] = admit_year - (demo["anchor_year"] - demo["anchor_age"])
    demo["age_at_first_admit"] = (
        demo["age_at_first_admit"].clip(lower=0, upper=120).fillna(0)
    )
    return demo[["subject_id", "gender", "age_at_first_admit"]]


def aggregate_race_insurance(admissions: pd.DataFrame) -> pd.DataFrame:
    agg = (
        admissions.groupby("subject_id")
        .agg(
            race=("race", mode_or_nan),
            insurance=("insurance", mode_or_nan),
            n_hospital_admits=("hadm_id", "nunique"),
            avg_los_days=("los", "mean"),
            total_los_days=("los", "sum"),
        )
        .reset_index()
    )
    for col in ["avg_los_days", "total_los_days"]:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)
    return agg


def compute_los_days(admissions: pd.DataFrame) -> pd.DataFrame:
    adm = admissions.copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    los_days = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400.0
    adm["los"] = los_days.clip(lower=0).fillna(0.0)
    return adm


OPIOID_PATTERNS = [
    "morphine", "hydromorphone", "oxycodone", "hydrocodone", "fentanyl",
    "codeine", "tramadol", "oxymorphone", "tapentadol", "methadone", "buprenorphine",
]

BENZO_PATTERNS = [
    "diazepam", "lorazepam", "alprazolam", "clonazepam", "temazepam",
    "chlordiazepoxide", "midazolam", "oxazepam",
]

# Simple ATC 1st-level mapping
ATC_MAP = {
    "antibiotic": "J", "vancomycin": "J", "ciprofloxacin": "J", "azithromycin": "J",
    "sertraline": "N", "risperidone": "N", "quetiapine": "N", "haloperidol": "N",
    "metoprolol": "C", "lisinopril": "C", "atorvastatin": "C", "amlodipine": "C", "furosemide": "C",
    "insulin": "A", "metformin": "A", "omeprazole": "A",
    "heparin": "B", "warfarin": "B",
    "albuterol": "R", "prednisone": "H",
}


def atc_group(drug: str) -> str:
    if not isinstance(drug, str):
        return "Other"
    drug_l = drug.lower()
    for key, atc in ATC_MAP.items():
        if key in drug_l:
            return atc
    return "Other"


def build_rx_features_atc(prescriptions: pd.DataFrame) -> pd.DataFrame:
    rx = prescriptions.copy()
    rx["is_opioid"] = rx["drug"].apply(
        lambda s: any(p in str(s).lower() for p in OPIOID_PATTERNS)
    )
    rx["is_benzo"] = rx["drug"].apply(
        lambda s: any(p in str(s).lower() for p in BENZO_PATTERNS)
    )
    
    opioid = rx[rx["is_opioid"]].copy()
    benzo = rx[rx["is_benzo"]].copy()
    other = rx[(~rx["is_opioid"]) & (~rx["is_benzo"])]
    
    # ATC grouping for "other"
    other["atc_group"] = other["drug"].apply(atc_group)
    atc_counts = other.groupby(["subject_id", "atc_group"]).size().unstack(fill_value=0)
    atc_counts.columns = [f"atc_{col}_rx_count" for col in atc_counts.columns]
    atc_counts = atc_counts.reset_index()

    def agg_rx(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[
                "subject_id", f"{kind}_rx_count", f"{kind}_hadms", 
                f"distinct_{kind}s", f"{kind}_exposure_days"
            ])
        df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
        df["stoptime"] = pd.to_datetime(df["stoptime"], errors="coerce")
        dur = (df["stoptime"] - df["starttime"]).dt.total_seconds() / 86400.0
        df["duration_days"] = np.maximum(0.0, np.nan_to_num(dur))
        
        g = df.groupby("subject_id").agg(**{
            f"{kind}_rx_count": ("drug", "count"),
            f"{kind}_hadms": ("hadm_id", "nunique"),
            f"distinct_{kind}s": ("drug", "nunique"),
            f"{kind}_exposure_days": ("duration_days", "sum"),
        }).reset_index()
        return g

    opioid_agg = agg_rx(opioid, "opioid")
    benzo_agg = agg_rx(benzo, "benzo")
    
    # Flags
    benzo_flag = (
        benzo.groupby("subject_id").size().reset_index(name="cnt")
        if not benzo.empty else pd.DataFrame(columns=["subject_id", "cnt"])
    )
    benzo_flag["any_benzo_flag"] = 1
    benzo_flag = benzo_flag[["subject_id", "any_benzo_flag"]]
    
    out = opioid_agg.merge(benzo_flag, on="subject_id", how="outer")
    out["any_benzo_flag"] = out["any_benzo_flag"].fillna(0).astype(int)
    out["any_opioid_flag"] = (out["opioid_rx_count"].fillna(0) > 0).astype(int)
    
    for col in ["opioid_rx_count", "opioid_hadms", "distinct_opioids", "opioid_exposure_days"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    
    # Merge in ATC class features
    out = out.merge(atc_counts, on="subject_id", how="left")
    for col in atc_counts.columns:
        if col != "subject_id":
            out[col] = out[col].fillna(0)
    return out


def build_oud_label(diagnoses: pd.DataFrame) -> pd.DataFrame:
    d = diagnoses.copy()
    d["icd_code"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    def is_oud(row) -> int:
        ver = int(row["icd_version"]) if not pd.isna(row["icd_version"]) else None
        code = row["icd_code"]
        if ver == 9:
            return int(code.startswith("3040") or code.startswith("3047") or code.startswith("3055"))
        elif ver == 10:
            return int(code.startswith("F11"))
        return 0

    d["is_oud"] = d.apply(is_oud, axis=1)
    lbl = d.groupby("subject_id")["is_oud"].max().reset_index().rename(columns={"is_oud": "y_oud"})
    return lbl


def get_rf_feature_importance(X: pd.DataFrame, y: pd.Series, target_name: str, top_n: int = 15) -> list:
    """Use Random Forest feature importance as proxy for SHAP"""
    print(f"\n=== Random Forest Feature Importance for {target_name} ===")
    
    if y.sum() < 2:
        print(f"Warning: {target_name} has only {y.sum()} positive cases. Skipping analysis.")
        return []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.sum() >= 2 else None
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} most important features for {target_name}:")
    print(feature_importance.head(top_n))

    return feature_importance.head(top_n)["feature"].tolist()


def main():
    print("=== Feature Selection with Proper Data Splitting (50K Synthetic Records) ===")

    # Load synthetic data
    DEMO_DIR = os.path.join("..", "synthetic_data", "mimic-clinical-iv-demo", "hosp")
    
    print("Loading synthetic data...")
    patients = read_csv("patients.csv.gz", DEMO_DIR)
    admissions_raw = read_csv("admissions.csv.gz", DEMO_DIR)
    diagnoses = read_csv("diagnoses_icd.csv.gz", DEMO_DIR)
    prescriptions = read_csv("prescriptions.csv.gz", DEMO_DIR)
    
    print(f"Loaded data shapes:")
    print(f"  Patients: {patients.shape}")
    print(f"  Admissions: {admissions_raw.shape}")
    print(f"  Diagnoses: {diagnoses.shape}")
    print(f"  Prescriptions: {prescriptions.shape}")

    print("\nProcessing raw data...")
    admissions = compute_los_days(admissions_raw)
    ri_util = aggregate_race_insurance(admissions)
    demo = compute_age_at_first_admit(patients, admissions)
    rx = build_rx_features_atc(prescriptions)
    y = build_oud_label(diagnoses)

    # Combine all features
    df = (demo.merge(ri_util, on="subject_id", how="left")
         .merge(rx, on="subject_id", how="left")
         .merge(y, on="subject_id", how="left"))

    # Fill missing values
    df["race"] = df["race"].fillna("UNKNOWN")
    df["insurance"] = df["insurance"].fillna("UNKNOWN")
    for col in ["n_hospital_admits", "avg_los_days", "total_los_days", "opioid_rx_count", 
                "opioid_hadms", "distinct_opioids", "opioid_exposure_days", "any_benzo_flag", "any_opioid_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["y_oud"] = df["y_oud"].fillna(0).astype(int)
    df["will_get_opioid_rx"] = df["any_opioid_flag"].fillna(0).astype(int)

    print(f"\nTotal records: {len(df):,}")
    print(f"Total features: {len(df.columns) - 3}")
    print(f"OUD positive cases: {df['y_oud'].sum():,} ({df['y_oud'].mean()*100:.1f}%)")
    print(f"Opioid prescription cases: {df['will_get_opioid_rx'].sum():,} ({df['will_get_opioid_rx'].mean()*100:.1f}%)")

    # *** CRITICAL: IMPLEMENT DATA SPLITTING ***
    print("\n" + "=" * 60)
    print("IMPLEMENTING PROPER DATA SPLITTING")
    print("=" * 60)
    
    # Strategy: 70% Neural Network, 30% Final Test, 7% Feature Selection from NN portion
    
    # First split: Separate final holdout test set (30%)
    df_dev, df_final_test = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['y_oud']
    )
    
    # Second split: From development set, take portion for feature selection (10% of dev = 7% of total)
    df_feature_selection, df_neural_network = train_test_split(
        df_dev, test_size=0.90, random_state=42, stratify=df_dev['y_oud']
    )
    
    print(f"Data split results:")
    print(f"  Feature selection subset: {len(df_feature_selection):,} records ({len(df_feature_selection)/len(df)*100:.1f}% of total)")
    print(f"  Neural Network training: {len(df_neural_network):,} records ({len(df_neural_network)/len(df)*100:.1f}% of total)")
    print(f"  Final holdout test: {len(df_final_test):,} records ({len(df_final_test)/len(df)*100:.1f}% of total)")

    # Prepare features for analysis
    def prepare_features(df_input):
        df_processed = df_input.copy()
        categorical_cols = ["gender", "race", "insurance"]
        label_encoders = {}

        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = df_processed[col].fillna("UNKNOWN")
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le

        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]:
                df_processed[col] = df_processed[col].fillna(0)

        feature_cols = [col for col in df_processed.columns 
                       if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]]
        X = df_processed[feature_cols]
        y_oud = df_processed["y_oud"].fillna(0)
        y_rx = df_processed["will_get_opioid_rx"].fillna(0)
        return X, y_oud, y_rx, feature_cols

    # Use only the feature selection subset for importance analysis
    print(f"\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    print(f"Using {len(df_feature_selection):,} records for feature selection...")

    X, y_oud, y_rx, feature_cols = prepare_features(df_feature_selection)

    # Get feature importance using Random Forest
    oud_features = get_rf_feature_importance(X, y_oud, "y_oud (Opioid Use Disorder)", top_n=15)
    rx_features = get_rf_feature_importance(X, y_rx, "will_get_opioid_rx (Opioid Prescriptions)", top_n=15)

    # Combine important features
    all_important_features = list(set(oud_features + rx_features))
    print(f"\nCombined important features: {len(all_important_features)}")
    print("Selected features:", all_important_features)

    # Save datasets for Neural Network training
    print(f"\n" + "=" * 60)
    print("SAVING DATASETS FOR NEURAL NETWORK")
    print("=" * 60)

    # Create final feature set
    final_features = ["subject_id"] + all_important_features + ["y_oud", "will_get_opioid_rx"]
    
    # Output directory for AI layer
    ai_layer_dir = os.path.join("..", "ai-layer")
    
    # Neural Network training set (with selected features)
    df_nn_train = df_neural_network[final_features].copy()
    df_nn_train.to_csv(os.path.join(ai_layer_dir, "neural_network_training_data.csv"), index=False)
    
    # Final test set (with selected features)
    df_final_test_selected = df_final_test[final_features].copy()
    df_final_test_selected.to_csv(os.path.join(ai_layer_dir, "neural_network_final_test_data.csv"), index=False)
    
    # Full dataset with selected features (for reference)
    df_full_selected = df[final_features].copy()
    df_full_selected.to_csv(os.path.join(ai_layer_dir, "full_dataset_selected_features.csv"), index=False)

    print(f"[SAVED] Neural Network training data: neural_network_training_data.csv")
    print(f"        Shape: {df_nn_train.shape}")
    print(f"[SAVED] Neural Network final test data: neural_network_final_test_data.csv")
    print(f"        Shape: {df_final_test_selected.shape}")
    print(f"[SAVED] Full dataset with selected features: full_dataset_selected_features.csv")
    print(f"        Shape: {df_full_selected.shape}")

    # Save feature importance results
    importance_df = pd.DataFrame({
        "feature": all_important_features,
        "selected_for_oud": [f in oud_features for f in all_important_features],
        "selected_for_rx": [f in rx_features for f in all_important_features],
    })
    importance_df.to_csv(os.path.join(ai_layer_dir, "feature_importance_results.csv"), index=False)
    print(f"[SAVED] Feature importance results: feature_importance_results.csv")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Original features: {len(df.columns) - 3}")
    print(f"✓ Selected features: {len(all_important_features)}")
    print(f"✓ Feature reduction: {((len(df.columns) - 3) - len(all_important_features)) / (len(df.columns) - 3) * 100:.1f}%")
    print(f"✓ Feature selection used: {len(df_feature_selection):,} records ({len(df_feature_selection)/len(df)*100:.1f}%)")
    print(f"✓ Neural Network training: {len(df_neural_network):,} records ({len(df_neural_network)/len(df)*100:.1f}%)")
    print(f"✓ Final holdout test: {len(df_final_test):,} records ({len(df_final_test)/len(df)*100:.1f}%)")
    print(f"✓ No data leakage between feature selection and model training!")
    print(f"✓ Ready for Neural Network development!")


if __name__ == "__main__":
    main()