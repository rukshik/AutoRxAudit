import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    "morphine",
    "hydromorphone",
    "oxycodone",
    "hydrocodone",
    "fentanyl",
    "codeine",
    "tramadol",
    "oxymorphone",
    "tapentadol",
    "methadone",
    "buprenorphine",
]

BENZO_PATTERNS = [
    "diazepam",
    "lorazepam",
    "alprazolam",
    "clonazepam",
    "temazepam",
    "chlordiazepoxide",
    "midazolam",
    "oxazepam",
]

# Simple ATC 1st-level mapping for demonstration
ATC_MAP = {
    "antibiotic": "J",  # Anti-infectives for systemic use
    "penicillin": "J",
    "cephalosporin": "J",
    "antidepressant": "N",  # Nervous system
    "ssri": "N",
    "snri": "N",
    "antipsychotic": "N",
    "anticonvulsant": "N",
    "antihypertensive": "C",  # Cardiovascular system
    "beta blocker": "C",
    "ace inhibitor": "C",
    "statin": "C",
    "insulin": "A",  # Alimentary tract and metabolism
    "metformin": "A",
    "proton pump inhibitor": "A",
    "prazole": "A",
    "anticoagulant": "B",  # Blood and blood forming organs
    "heparin": "B",
    "warfarin": "B",
    "antiplatelet": "B",
    "bronchodilator": "R",  # Respiratory system
    "albuterol": "R",
    "inhaler": "R",
    "thyroid": "H",  # Systemic hormonal preparations
    "levothyroxine": "H",
    "glucocorticoid": "H",
    "prednisone": "H",
    # Add more as needed
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

    # Aggregate as before
    def agg_rx(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "subject_id",
                    f"{kind}_rx_count",
                    f"{kind}_hadms",
                    f"distinct_{kind}s",
                    f"{kind}_exposure_days",
                ]
            )
        df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
        df["stoptime"] = pd.to_datetime(df["stoptime"], errors="coerce")
        dur = (df["stoptime"] - df["starttime"]).dt.total_seconds() / 86400.0
        df["duration_days"] = np.maximum(0.0, np.nan_to_num(dur))
        g = (
            df.groupby("subject_id")
            .agg(
                **{
                    f"{kind}_rx_count": ("drug", "count"),
                    f"{kind}_hadms": ("hadm_id", "nunique"),
                    f"distinct_{kind}s": ("drug", "nunique"),
                    f"{kind}_exposure_days": ("duration_days", "sum"),
                }
            )
            .reset_index()
        )
        return g

    opioid_agg = agg_rx(opioid, "opioid")
    benzo_agg = agg_rx(benzo, "benzo")
    # Flags
    benzo_flag = (
        benzo.groupby("subject_id").size().reset_index(name="cnt")
        if not benzo.empty
        else pd.DataFrame(columns=["subject_id", "cnt"])
    )
    benzo_flag["any_benzo_flag"] = 1
    benzo_flag = benzo_flag[["subject_id", "any_benzo_flag"]]
    out = opioid_agg.merge(benzo_flag, on="subject_id", how="outer")
    out["any_benzo_flag"] = out["any_benzo_flag"].fillna(0).astype(int)
    out["any_opioid_flag"] = (out["opioid_rx_count"].fillna(0) > 0).astype(int)
    for col in [
        "opioid_rx_count",
        "opioid_hadms",
        "distinct_opioids",
        "opioid_exposure_days",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    # Merge in ATC class features
    out = out.merge(atc_counts, on="subject_id", how="left")
    # Fill missing ATC columns with 0
    for col in atc_counts.columns:
        if col != "subject_id":
            out[col] = out[col].fillna(0)
    return out


def build_oud_label(diagnoses: pd.DataFrame) -> pd.DataFrame:
    d = diagnoses.copy()
    if "icd_version" not in d.columns or "icd_code" not in d.columns:
        raise ValueError(
            "Expected columns 'icd_version' and 'icd_code' in diagnoses_icd.csv.gz"
        )
    d["icd_code"] = (
        d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
    )

    def is_oud(row) -> int:
        ver = int(row["icd_version"]) if not pd.isna(row["icd_version"]) else None
        code = row["icd_code"]
        if ver == 9:
            return int(
                code.startswith("3040")
                or code.startswith("3047")
                or code.startswith("3055")
            )
        elif ver == 10:
            return int(code.startswith("F11"))
        return 0

    d["is_oud"] = d.apply(is_oud, axis=1)
    lbl = (
        d.groupby("subject_id")["is_oud"]
        .max()
        .reset_index()
        .rename(columns={"is_oud": "y_oud"})
    )
    return lbl


def prepare_features_for_shap(df: pd.DataFrame) -> tuple:
    """
    Prepare features for SHAP analysis by encoding categorical variables
    and handling missing values.
    """
    # Create a copy for processing
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
        col
        for col in df_processed.columns
        if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]
    ]

    X = df_processed[feature_cols]
    y_oud = df_processed["y_oud"].fillna(0)
    y_rx = df_processed["will_get_opioid_rx"].fillna(0)

    return X, y_oud, y_rx, feature_cols, label_encoders


def get_shap_feature_importance(
    X: pd.DataFrame, y: pd.Series, target_name: str, top_n: int = 10
) -> list:
    """
    Use SHAP to identify the most important features for a given target.
    """
    print(f"\n=== SHAP Analysis for {target_name} ===")

    # Check if target has enough positive cases
    if y.sum() < 2:
        print(
            f"Warning: {target_name} has only {y.sum()} positive cases. Skipping SHAP analysis."
        )
        return []

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.sum() >= 2 else None
    )

    # Train Random Forest model
    print(f"Training Random Forest model for {target_name}...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate SHAP values
    print(f"Calculating SHAP values for {target_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification (use positive class SHAP values)
    if len(shap_values) == 2:
        shap_values = shap_values[1]  # Use positive class

    # Calculate mean absolute SHAP values for feature importance
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "shap_importance": mean_shap_values}
    ).sort_values("shap_importance", ascending=False)

    print(f"\nTop {top_n} most important features for {target_name}:")
    print(feature_importance.head(top_n))

    # Return top N features
    top_features = feature_importance.head(top_n)["feature"].tolist()
    return top_features


def main():
    print("=== SHAP-Based Feature Selection for Opioid Prediction ===")

    # Load and process data (same as original)
    DEMO_DIR = os.path.join("data", "mimic-clinical-iv-demo", "hosp")
    patients = read_csv("patients.csv.gz", DEMO_DIR)
    admissions_raw = read_csv("admissions.csv.gz", DEMO_DIR)
    diagnoses = read_csv("diagnoses_icd.csv.gz", DEMO_DIR)
    prescriptions = read_csv("prescriptions.csv.gz", DEMO_DIR)

    print("Processing raw data...")
    admissions = compute_los_days(admissions_raw)
    ri_util = aggregate_race_insurance(admissions)
    demo = compute_age_at_first_admit(patients, admissions)
    rx = build_rx_features_atc(prescriptions)
    y = build_oud_label(diagnoses)

    # Combine all features
    df = (
        demo.merge(ri_util, on="subject_id", how="left")
        .merge(rx, on="subject_id", how="left")
        .merge(y, on="subject_id", how="left")
    )

    # Fill missing values
    df["race"] = df["race"].fillna("UNKNOWN")
    df["insurance"] = df["insurance"].fillna("UNKNOWN")
    for col in [
        "n_hospital_admits",
        "avg_los_days",
        "total_los_days",
        "opioid_rx_count",
        "opioid_hadms",
        "distinct_opioids",
        "opioid_exposure_days",
        "any_benzo_flag",
        "any_opioid_flag",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["y_oud"] = df["y_oud"].fillna(0).astype(int)
    df["will_get_opioid_rx"] = df["any_opioid_flag"].fillna(0).astype(int)

    print(
        f"Total features before SHAP selection: {len(df.columns) - 3}"
    )  # -3 for subject_id and targets

    # Prepare features for SHAP analysis
    X, y_oud, y_rx, feature_cols, label_encoders = prepare_features_for_shap(df)

    # Get SHAP-based feature importance for both targets
    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Analyze y_oud (if sufficient data)
    oud_features = get_shap_feature_importance(
        X, y_oud, "y_oud (Opioid Use Disorder)", top_n=10
    )

    # Analyze will_get_opioid_rx
    rx_features = get_shap_feature_importance(
        X, y_rx, "will_get_opioid_rx (Will receive opioid prescription)", top_n=15
    )

    # Combine important features from both analyses
    all_important_features = list(set(oud_features + rx_features))
    print(f"\nCombined important features: {len(all_important_features)}")
    print("Features:", all_important_features)

    # Create final dataset with only important features
    final_features = (
        ["subject_id"] + all_important_features + ["y_oud", "will_get_opioid_rx"]
    )
    df_final = df[final_features].copy()

    # Save the SHAP-selected features
    out_csv = "mimiciv_demo_oud_features_with_shap_selection.csv"
    df_final.to_csv(out_csv, index=False)

    print(f"\n=== RESULTS ===")
    print(f"[OK] Wrote SHAP-selected features: {out_csv}")
    print(f"Original features: {len(df.columns) - 3}")
    print(f"SHAP-selected features: {len(all_important_features)}")
    print(
        f"Reduction: {((len(df.columns) - 3) - len(all_important_features)) / (len(df.columns) - 3) * 100:.1f}%"
    )
    print(f"\nFinal dataset shape: {df_final.shape}")
    print("\nFirst few rows:")
    print(df_final.head())

    # Save feature importance results
    importance_csv = "shap_feature_importance_results.csv"
    importance_df = pd.DataFrame(
        {
            "feature": all_important_features,
            "selected_for_oud": [f in oud_features for f in all_important_features],
            "selected_for_rx": [f in rx_features for f in all_important_features],
        }
    )
    importance_df.to_csv(importance_csv, index=False)
    print(f"[OK] Wrote feature importance results: {importance_csv}")


if __name__ == "__main__":
    main()
