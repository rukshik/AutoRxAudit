"""
SHAP-Based Feature Selection for OUD Prediction
Reads synthetic data, performs SHAP analysis, and saves processed data for main model
"""

import os
import sys
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def read_csv(rel_path: str, demo_dir: str) -> pd.DataFrame:
    """Read CSV file from the given directory"""
    path = os.path.join(demo_dir, rel_path)
    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def mode_or_nan(series: pd.Series):
    """Return mode of series or NaN if empty"""
    if series.empty:
        return np.nan
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.idxmax()


def compute_age_at_first_admit(
    patients: pd.DataFrame, admissions: pd.DataFrame
) -> pd.DataFrame:
    """Calculate patient age at first admission"""
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
    """Aggregate admission-level race/insurance and calculate hospital utilization"""
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
    """Calculate length of stay in days"""
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

# ATC (Anatomical Therapeutic Chemical) classification mapping
ATC_MAP = {
    "antibiotic": "J", "penicillin": "J", "cephalosporin": "J", "vancomycin": "J",
    "ciprofloxacin": "J", "azithromycin": "J", "amoxicillin": "J",
    "antidepressant": "N", "ssri": "N", "snri": "N", "sertraline": "N",
    "antipsychotic": "N", "risperidone": "N", "quetiapine": "N", "haloperidol": "N",
    "anticonvulsant": "N",
    "antihypertensive": "C", "beta blocker": "C", "metoprolol": "C", "carvedilol": "C",
    "ace inhibitor": "C", "lisinopril": "C", "statin": "C", "atorvastatin": "C",
    "simvastatin": "C", "amlodipine": "C", "furosemide": "C",
    "insulin": "A", "metformin": "A", "glipizide": "A", "lantus": "A",
    "proton pump inhibitor": "A", "prazole": "A", "omeprazole": "A",
    "anticoagulant": "B", "heparin": "B", "warfarin": "B", "antiplatelet": "B",
    "bronchodilator": "R", "albuterol": "R", "inhaler": "R",
    "thyroid": "H", "levothyroxine": "H", "glucocorticoid": "H", "prednisone": "H",
}

ATC_LABELS = {
    "A": "Alimentary tract/metabolism",
    "B": "Blood/organs",
    "C": "Cardiovascular",
    "H": "Hormonal",
    "J": "Anti-infectives",
    "N": "Nervous system",
    "R": "Respiratory",
}


def atc_group(drug: str) -> str:
    """Map drug name to ATC category"""
    if not isinstance(drug, str):
        return "Other"
    drug_l = drug.lower()
    for key, atc in ATC_MAP.items():
        if key in drug_l:
            return atc
    return "Other"


def build_rx_features_atc(prescriptions: pd.DataFrame) -> pd.DataFrame:
    """Build prescription features including opioid, benzo, and ATC categories"""
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
    
    # ATC grouping for "other" drugs
    other["atc_group"] = other["drug"].apply(atc_group)
    atc_counts = other.groupby(["subject_id", "atc_group"]).size().unstack(fill_value=0)
    atc_counts.columns = [f"atc_{col}_rx_count" for col in atc_counts.columns]
    atc_counts = atc_counts.reset_index()

    def agg_rx(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Aggregate prescription data by patient"""
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
    
    # Create flags
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
    
    # Merge ATC class features
    out = out.merge(atc_counts, on="subject_id", how="left")
    for col in atc_counts.columns:
        if col != "subject_id":
            out[col] = out[col].fillna(0)
    return out


def build_oud_label(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Create OUD label from ICD diagnosis codes"""
    d = diagnoses.copy()
    d["icd_code"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    def is_oud(row) -> int:
        ver = int(row["icd_version"]) if not pd.isna(row["icd_version"]) else None
        code = row["icd_code"]
        if ver == 9:
            # ICD-9: 304.0x (opioid dependence), 304.7x (combinations), 305.5x (abuse)
            return int(code.startswith("3040") or code.startswith("3047") or code.startswith("3055"))
        elif ver == 10:
            # ICD-10: F11.x (opioid-related disorders)
            return int(code.startswith("F11"))
        return 0

    d["is_oud"] = d.apply(is_oud, axis=1)
    lbl = d.groupby("subject_id")["is_oud"].max().reset_index().rename(columns={"is_oud": "y_oud"})
    return lbl


def prepare_features_for_shap(df: pd.DataFrame) -> tuple:
    """Encode categorical variables and prepare features for SHAP analysis"""
    df_processed = df.copy()

    # Handle categorical variables
    categorical_cols = ["gender", "race", "insurance"]
    label_encoders = {}

    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna("UNKNOWN")
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le

    # Fill missing values for numeric columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]:
            df_processed[col] = df_processed[col].fillna(0)

    # Separate features and targets
    feature_cols = [
        col for col in df_processed.columns
        if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]
    ]

    X = df_processed[feature_cols]
    y_oud = df_processed["y_oud"].fillna(0)
    y_rx = df_processed["will_get_opioid_rx"].fillna(0)

    return X, y_oud, y_rx, feature_cols, label_encoders


def get_shap_feature_importance(
    X: pd.DataFrame, y: pd.Series, target_name: str, top_n: int = 15
) -> list:
    """Use SHAP to identify the most important features for a given target"""
    print(f"\n=== SHAP Analysis for {target_name} ===")

    # Check if target has enough positive cases
    if y.sum() < 2:
        print(f"Warning: {target_name} has only {y.sum()} positive cases. Skipping SHAP analysis.")
        return []

    # Split data for model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.sum() >= 2 else None
    )

    # Train Random Forest model
    print(f"Training Random Forest model for {target_name}...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Model accuracy - Train: {train_acc:.3f}, Test: {test_acc:.3f}")

    # Calculate SHAP values
    print(f"Calculating SHAP values for {target_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification (use positive class SHAP values)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # Use positive class

    # Calculate mean absolute SHAP values for feature importance
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "shap_importance": mean_shap_values
    }).sort_values("shap_importance", ascending=False)

    print(f"\nTop {top_n} most important features for {target_name}:")
    print(feature_importance.head(top_n).to_string(index=False))

    # Return top N features
    top_features = feature_importance.head(top_n)["feature"].tolist()
    return top_features, feature_importance


def main():
    print("=" * 80)
    print("SHAP-BASED FEATURE SELECTION FOR OUD PREDICTION")
    print("=" * 80)

    # Define paths
    SYNTHETIC_DATA_DIR = os.path.join("..", "..", "synthetic_data", "mimic-clinical-iv-demo", "hosp")
    INTERMEDIATE_DIR = os.path.join("temp_data")
    PROCESSED_DATA_DIR = os.path.join("..", "processed_data")

    # Load synthetic data
    print("\n[1/6] Loading synthetic data...")
    patients = read_csv("patients.csv.gz", SYNTHETIC_DATA_DIR)
    admissions_raw = read_csv("admissions.csv.gz", SYNTHETIC_DATA_DIR)
    diagnoses = read_csv("diagnoses_icd.csv.gz", SYNTHETIC_DATA_DIR)
    prescriptions = read_csv("prescriptions.csv.gz", SYNTHETIC_DATA_DIR)
    
    print(f"  Patients: {patients.shape[0]:,} records")
    print(f"  Admissions: {admissions_raw.shape[0]:,} records")
    print(f"  Diagnoses: {diagnoses.shape[0]:,} records")
    print(f"  Prescriptions: {prescriptions.shape[0]:,} records")

    # Process and engineer features
    print("\n[2/6] Engineering features...")
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
                "opioid_hadms", "distinct_opioids", "opioid_exposure_days", "any_benzo_flag", 
                "any_opioid_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["y_oud"] = df["y_oud"].fillna(0).astype(int)
    df["will_get_opioid_rx"] = df["any_opioid_flag"].fillna(0).astype(int)

    print(f"  Total patients: {len(df):,}")
    print(f"  Total features (before selection): {len(df.columns) - 3}")
    print(f"  OUD positive cases: {df['y_oud'].sum():,} ({df['y_oud'].mean()*100:.1f}%)")
    print(f"  Opioid prescription cases: {df['will_get_opioid_rx'].sum():,} ({df['will_get_opioid_rx'].mean()*100:.1f}%)")

    # Data splitting strategy
    print("\n[3/6] Splitting data for proper ML workflow...")
    print("  Strategy: 7% feature selection, 56% training, 7% validation, 30% test")
    
    # First split: Separate final holdout test set (30%)
    df_dev, df_test = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['y_oud']
    )
    
    # Second split: From development set, take portion for SHAP (10% of dev = 7% of total)
    df_shap, df_remaining = train_test_split(
        df_dev, test_size=0.90, random_state=42, stratify=df_dev['y_oud']
    )
    
    # Third split: Split remaining into train and validation (90% train, 10% val from remaining)
    df_train, df_val = train_test_split(
        df_remaining, test_size=0.10, random_state=42, stratify=df_remaining['y_oud']
    )
    
    print(f"  SHAP analysis subset: {len(df_shap):,} records ({len(df_shap)/len(df)*100:.1f}%)")
    print(f"  Model training set: {len(df_train):,} records ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Model validation set: {len(df_val):,} records ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Final test set: {len(df_test):,} records ({len(df_test)/len(df)*100:.1f}%)")

    # Save intermediate data splits
    df_shap.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_subset.csv"), index=False)
    print(f"  [Saved] SHAP subset to temp_data/shap_subset.csv")

    # SHAP feature importance analysis
    print("\n[4/6] Performing SHAP analysis on subset...")
    X, y_oud, y_rx, feature_cols, label_encoders = prepare_features_for_shap(df_shap)

    # Analyze both targets (get all features ranked)
    oud_features_all, oud_importance = get_shap_feature_importance(
        X, y_oud, "y_oud (Opioid Use Disorder)", top_n=25
    )

    rx_features_all, rx_importance = get_shap_feature_importance(
        X, y_rx, "will_get_opioid_rx (Opioid Prescription)", top_n=25
    )

    # Analyze importance distribution to find natural cutoffs
    print("\n  Analyzing SHAP importance distribution...")
    print("  OUD feature importance percentiles:")
    percentiles = [50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(oud_importance['shap_importance'], p)
        count = (oud_importance['shap_importance'] >= val).sum()
        print(f"    {p}th percentile: {val:.6f} ({count} features)")
    
    print("  RX feature importance percentiles:")
    for p in percentiles:
        val = np.percentile(rx_importance['shap_importance'], p)
        count = (rx_importance['shap_importance'] >= val).sum()
        print(f"    {p}th percentile: {val:.6f} ({count} features)")
    
    # Select features above 50th percentile for each target
    oud_threshold = np.percentile(oud_importance['shap_importance'], 50)
    rx_threshold = np.percentile(rx_importance['shap_importance'], 50)
    
    oud_features = oud_importance[oud_importance['shap_importance'] >= oud_threshold]['feature'].tolist()
    rx_features = rx_importance[rx_importance['shap_importance'] >= rx_threshold]['feature'].tolist()
    
    print(f"\n  Features selected using 50th percentile threshold:")
    print(f"    OUD: {len(oud_features)} features (threshold: {oud_threshold:.6f})")
    print(f"    RX: {len(rx_features)} features (threshold: {rx_threshold:.6f})")

    # Combine important features from both analyses
    all_important_features = list(set(oud_features + rx_features))
    print(f"\n  Combined unique important features: {len(all_important_features)}")
    print(f"  Feature reduction: {len(feature_cols)} → {len(all_important_features)} "
          f"({(len(feature_cols) - len(all_important_features)) / len(feature_cols) * 100:.1f}% reduction)")

    # Save SHAP importance results
    importance_df = pd.DataFrame({
        "feature": all_important_features,
        "selected_for_oud": [f in oud_features for f in all_important_features],
        "selected_for_rx": [f in rx_features for f in all_important_features],
    })
    importance_df.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_feature_importance.csv"), index=False)
    
    # Save detailed importance scores
    oud_importance.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_importance_oud.csv"), index=False)
    rx_importance.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_importance_rx.csv"), index=False)
    print(f"  [Saved] Feature importance results to temp_data/")

    # Prepare final datasets with selected features
    print("\n[5/6] Creating processed datasets with selected features...")
    final_features = ["subject_id"] + all_important_features + ["y_oud", "will_get_opioid_rx"]
    
    # Training set (with selected features only)
    df_train_processed = df_train[final_features].copy()
    df_train_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_data.csv"), index=False)
    print(f"  [Saved] Training data: processed_data/train_data.csv ({df_train_processed.shape})")
    
    # Validation set (with selected features only)
    df_val_processed = df_val[final_features].copy()
    df_val_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "validation_data.csv"), index=False)
    print(f"  [Saved] Validation data: processed_data/validation_data.csv ({df_val_processed.shape})")
    
    # Test set (with selected features only)
    df_test_processed = df_test[final_features].copy()
    df_test_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_data.csv"), index=False)
    print(f"  [Saved] Test data: processed_data/test_data.csv ({df_test_processed.shape})")
    
    # Full dataset with selected features (for reference)
    df_full_processed = df[final_features].copy()
    df_full_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "full_data_selected_features.csv"), index=False)
    print(f"  [Saved] Full dataset: processed_data/full_data_selected_features.csv ({df_full_processed.shape})")

    # Save metadata
    print("\n[6/6] Saving metadata and summary...")
    metadata = {
        "total_records": len(df),
        "train_records": len(df_train),
        "validation_records": len(df_val),
        "test_records": len(df_test),
        "shap_records": len(df_shap),
        "original_features": len(feature_cols),
        "selected_features": len(all_important_features),
        "feature_names": all_important_features,
        "oud_positive_train": int(df_train['y_oud'].sum()),
        "oud_positive_validation": int(df_val['y_oud'].sum()),
        "oud_positive_test": int(df_test['y_oud'].sum()),
        "data_split_strategy": "7% SHAP, 56% train, 7% validation, 30% test"
    }
    
    import json
    with open(os.path.join(PROCESSED_DATA_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [Saved] Metadata: processed_data/metadata.json")

    # Print final summary
    print("\n" + "=" * 80)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 80)
    print(f"✓ Original features: {len(feature_cols)}")
    print(f"✓ Selected features: {len(all_important_features)}")
    print(f"✓ SHAP analysis used: {len(df_shap):,} records ({len(df_shap)/len(df)*100:.1f}%)")
    print(f"✓ Model training data: {len(df_train):,} records ({len(df_train)/len(df)*100:.1f}%)")
    print(f"✓ Model validation data: {len(df_val):,} records ({len(df_val)/len(df)*100:.1f}%)")
    print(f"✓ Final test data: {len(df_test):,} records ({len(df_test)/len(df)*100:.1f}%)")
    print(f"✓ No data leakage between feature selection and model training")
    print(f"\nSelected features:")
    for i, feat in enumerate(sorted(all_important_features), 1):
        oud_mark = "✓" if feat in oud_features else " "
        rx_mark = "✓" if feat in rx_features else " "
        print(f"  {i:2d}. {feat:30s} [OUD:{oud_mark}] [RX:{rx_mark}]")
    print(f"\nProcessed data ready in: processed_data/")
    print(f"  - train_data.csv ({df_train_processed.shape[0]:,} records)")
    print(f"  - validation_data.csv ({df_val_processed.shape[0]:,} records)")
    print(f"  - test_data.csv ({df_test_processed.shape[0]:,} records)")
    print(f"  - metadata.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
