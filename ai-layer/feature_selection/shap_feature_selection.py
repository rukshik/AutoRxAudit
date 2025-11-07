"""
SHAP-Based Feature Selection

Usage:
  python shap_feature_selection.py
  python shap_feature_selection.py --input-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp
  python shap_feature_selection.py --output-dir ../processed_data_100k --temp-dir temp_data_100k
"""

from email import parser
import os
import sys

# SHAP is giving trouble with PyTorch, but we don't need it. So mocking
class MockTorch:
    def __getattr__(self, name):
        raise ImportError("PyTorch not available (mocked for feature selection)")

sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MockTorch()
sys.modules['torch.utils'] = MockTorch()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import argparse

from shap import TreeExplainer

warnings.filterwarnings("ignore")

# read csv file from a directory onto pandas
def read_csv(rel_path, dir_path):
    path = os.path.join(dir_path, rel_path)
    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)
    return pd.read_csv(path)

# Mode of a series or Numphy.Nan if empty
def mode_or_nan(series):
    if series.empty:
        return np.nan
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.idxmax()

# Age at first hospital admission 
def compute_age_at_first_admit(
    patients, admissions
):
    first_admit = (
        admissions.sort_values("admittime")
        .groupby("subject_id", as_index=False)
        .first()[["subject_id", "hadm_id", "admittime"]]
    )
    hosp = patients.merge(first_admit, on="subject_id", how="left")
    hosp["admittime"] = pd.to_datetime(hosp["admittime"], errors="coerce")
    admit_year = hosp["admittime"].dt.year
    hosp["age_at_first_admit"] = admit_year - (hosp["anchor_year"] - hosp["anchor_age"])
    hosp["age_at_first_admit"] = (
        hosp["age_at_first_admit"].clip(lower=0, upper=120).fillna(0)
    )
    return hosp[["subject_id", "gender", "age_at_first_admit"]]


# get race, insurance, number of hostpital admissions, and averge and total length of hospital stays
def aggregate_race_insurance(admissions):
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


# caculate length of hospital stays
def compute_los_days(admissions):
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

# Map a drug to ATC group
def atc_group(drug):
    if not isinstance(drug, str):
        return "Other"
    drug_l = drug.lower()
    for key, atc in ATC_MAP.items():
        if key in drug_l:
            return atc
    return "Other"


# build prescription drug feature set
def build_rx_features_atc(prescriptions):

    # Opioids
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
    
    # "Other" drugs
    other["atc_group"] = other["drug"].apply(atc_group)
    atc_counts = other.groupby(["subject_id", "atc_group"]).size().unstack(fill_value=0)
    atc_counts.columns = [f"atc_{col}_rx_count" for col in atc_counts.columns]
    atc_counts = atc_counts.reset_index()

    # prescription data by patient
    def agg_rx(df, kind):
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


# build oud label based on dignosis codes
def build_oud_label(diagnoses):
    d = diagnoses.copy()
    d["icd_code"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    def is_oud(row):
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


def build_opioid_eligibility_label(diagnoses):
    """
    Create opioid eligibility label based on pain-related diagnoses.
    Patient is eligible (=1) if they have any pain-related condition that would justify opioid use.
    This is based on the pain diagnosis codes from MIMIC data analysis.
    """
    # Load pain diagnosis codes
    pain_codes_file = os.path.join("..", "..", "data-processing", "data_generation", "pain_diagnosis_codes.json")
    if not os.path.exists(pain_codes_file):
        print(f"  Warning: Pain codes file not found at {pain_codes_file}")
        print(f"  Creating eligibility label with empty pain codes...")
        pain_icd9 = []
        pain_icd10 = []
    else:
        import json
        with open(pain_codes_file, 'r') as f:
            pain_codes = json.load(f)
            # Convert to sets for O(1) lookup instead of O(n)
            pain_icd9 = set(pain_codes['icd9'])
            pain_icd10 = set(pain_codes['icd10'])
        print(f"  Loaded {len(pain_icd9)} ICD-9 and {len(pain_icd10)} ICD-10 pain codes for eligibility")
    
    print(f"  → Checking {len(diagnoses):,} diagnoses against pain codes...", end=" ", flush=True)
    d = diagnoses.copy()
    d["icd_code"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
    
    # Vectorized approach: much faster than row-by-row apply
    # Check ICD-9 codes
    mask_icd9 = d["icd_version"] == 9
    d.loc[mask_icd9, "is_pain"] = d.loc[mask_icd9, "icd_code"].isin(pain_icd9).astype(int)
    
    # Check ICD-10 codes
    mask_icd10 = d["icd_version"] == 10
    d.loc[mask_icd10, "is_pain"] = d.loc[mask_icd10, "icd_code"].isin(pain_icd10).astype(int)
    
    # Fill any remaining NaN (other versions) with 0
    d["is_pain"] = d["is_pain"].fillna(0).astype(int)
    print("✓")
    

    # Patient is eligible if they have ANY pain diagnosis across all admissions
    eligibility = d.groupby("subject_id")["is_pain"].max().reset_index()
    eligibility = eligibility.rename(columns={"is_pain": "opioid_eligibility"})
    
    return eligibility


def prepare_features_for_shap(df):
    df_processed = df.copy()

    # encode gender, race and insurance for consistancy 
    categorical_cols = ["gender", "race", "insurance"]
    label_encoders = {}

    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = df_processed[col].fillna("UNKNOWN")
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le

    # missing numerical values with zero
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["subject_id", "y_oud", "will_get_opioid_rx"]:
            df_processed[col] = df_processed[col].fillna(0)

    # targets vs training features
    feature_cols = [
        col for col in df_processed.columns
        if col not in ["subject_id", "y_oud", "will_get_opioid_rx", "opioid_eligibility"]
    ]

    X = df_processed[feature_cols]
    y_oud = df_processed["y_oud"].fillna(0)
    y_rx = df_processed["will_get_opioid_rx"].fillna(0)
    y_eligibility = df_processed["opioid_eligibility"].fillna(0)

    return X, y_oud, y_rx, y_eligibility, feature_cols, label_encoders


# have SHAP caculate feature importance - called separatly for eligibility and oud
def get_shap_feature_importance(X, y, target_name, top_n=15):

    # Check if target has enough positive cases
    if y.sum() < 2:
        print(f"Warning: {target_name} has only {y.sum()} positive cases. Skipping SHAP analysis.")
        return []

    # Split data for model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.sum() >= 2 else None
    )

    # Train Random Forest model, which is typical for SHAP analysis
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # Calculate SHAP values. TreeExplainer does it.
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification (use positive class SHAP values)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # Use positive class

    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "shap_importance": mean_shap_values
    }).sort_values("shap_importance", ascending=False)

    top_features = feature_importance.head(top_n)["feature"].tolist()
    return top_features, feature_importance

# online medical records for BMI etc
def build_omr_features(omr):
    if omr is None or omr.empty:
        return pd.DataFrame(columns=["subject_id"])
    
    bmi_records = omr[omr["result_name"] == "BMI (kg/m2)"].copy()
    bmi_records["bmi_value"] = pd.to_numeric(bmi_records["result_value"], errors="coerce")
    
    # Most recent
    bmi_agg = bmi_records.groupby("subject_id").agg(
        bmi=("bmi_value", "mean"),
        has_bmi=("bmi_value", "count")
    ).reset_index()
    
    # caetegorize - use general value
    bmi_agg["obesity_flag"] = (bmi_agg["bmi"] > 30).astype(int)
    bmi_agg["morbid_obesity_flag"] = (bmi_agg["bmi"] > 40).astype(int)
    bmi_agg["has_bmi"] = (bmi_agg["has_bmi"] > 0).astype(int)
    
    return bmi_agg

# drug data
def build_drg_features(drgcodes):
    if drgcodes is None or drgcodes.empty:
        return pd.DataFrame(columns=["subject_id"])
    
    # Aggregate DRG features per patient
    drg_agg = drgcodes.groupby("subject_id").agg(
        avg_drg_severity=("drg_severity", "mean"),
        max_drg_severity=("drg_severity", "max"),
        avg_drg_mortality=("drg_mortality", "mean"),
        max_drg_mortality=("drg_mortality", "max"),
        n_admissions_with_drg=("hadm_id", "nunique")
    ).reset_index()
    
    # Create high severity flag (severity >= 3)
    drg_agg["high_severity_flag"] = (drg_agg["max_drg_severity"] >= 3).astype(int)
    drg_agg["high_mortality_flag"] = (drg_agg["max_drg_mortality"] >= 3).astype(int)
    
    return drg_agg

# icu data like stays and transfers, outcomes
def build_icu_features(transfers):
    if transfers is None or transfers.empty:
        return pd.DataFrame(columns=["subject_id"])
    
    # Identify ICU transfers
    icu_transfers = transfers[transfers["careunit"].str.contains("ICU", case=False, na=False)].copy()
    
    # Calculate ICU stay duration
    if not icu_transfers.empty:
        icu_transfers["intime"] = pd.to_datetime(icu_transfers["intime"], errors="coerce")
        icu_transfers["outtime"] = pd.to_datetime(icu_transfers["outtime"], errors="coerce")
        icu_transfers["icu_stay_hours"] = (
            (icu_transfers["outtime"] - icu_transfers["intime"]).dt.total_seconds() / 3600.0
        )
    
    # Aggregate ICU features per patient
    icu_agg = icu_transfers.groupby("subject_id").agg(
        n_icu_stays=("transfer_id", "count"),
        total_icu_hours=("icu_stay_hours", "sum"),
        n_icu_admissions=("hadm_id", "nunique")
    ).reset_index()
    
    # Create ICU flags
    icu_agg["has_icu_stay"] = (icu_agg["n_icu_stays"] > 0).astype(int)
    icu_agg["total_icu_days"] = icu_agg["total_icu_hours"] / 24.0
    
    # Add flag for patients with NO ICU stays
    all_patients = pd.DataFrame({"subject_id": transfers["subject_id"].unique()})
    icu_agg = all_patients.merge(icu_agg, on="subject_id", how="left")
    icu_agg = icu_agg.fillna(0)
    
    return icu_agg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', '-i', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    parser.add_argument('--temp-dir', '-t', type=str)
        
    args = parser.parse_args()
    
    SYNTHETIC_DATA_DIR = args.input_dir
    INTERMEDIATE_DIR = args.temp_dir
    PROCESSED_DATA_DIR = args.output_dir
    
    # Create output directories, if they don't exist
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Load  data nto pandas
    print("Reading patients.csv.gz...")
    patients = read_csv("patients.csv.gz", SYNTHETIC_DATA_DIR)
    
    print("Reading admissions.csv.gz...")
    admissions_raw = read_csv("admissions.csv.gz", SYNTHETIC_DATA_DIR)
    
    print("Reading diagnoses_icd.csv.gz...")
    diagnoses = read_csv("diagnoses_icd.csv.gz", SYNTHETIC_DATA_DIR)
    
    print("Reading prescriptions.csv.gz...")
    prescriptions = read_csv("prescriptions.csv.gz", SYNTHETIC_DATA_DIR)

    print("Reading omr.csv.gz (BMI data)...")
    omr = read_csv("omr.csv.gz", SYNTHETIC_DATA_DIR)

    print("Reading drgcodes.csv.gz (DRG severity)...")
    drgcodes = read_csv("drgcodes.csv.gz", SYNTHETIC_DATA_DIR)

    print("Reading transfers.csv.gz (ICU stays)...")
    transfers = read_csv("transfers.csv.gz", SYNTHETIC_DATA_DIR)
    
    print(f" Patients: {patients.shape[0]:,} records")
    print(f"Admissions: {admissions_raw.shape[0]:,} records")
    print(f"Diagnoses: {diagnoses.shape[0]:,} records")
    print(f"Prescriptions: {prescriptions.shape[0]:,} records")
    print(f"OMR: {omr.shape[0]:,} records (BMI data)")
    print(f"DRG Codes: {drgcodes.shape[0]:,} records")
    print(f"Transfers: {transfers.shape[0]:,} records")

    # Process features

    admissions = compute_los_days(admissions_raw)
    ri_util = aggregate_race_insurance(admissions)
    demo = compute_age_at_first_admit(patients, admissions)
    rx = build_rx_features_atc(prescriptions)
    y_oud = build_oud_label(diagnoses)
    y_eligibility = build_opioid_eligibility_label(diagnoses)
    bmi_features = build_omr_features(omr)
    drg_features = build_drg_features(drgcodes)
    icu_features = build_icu_features(transfers)

    # Combine all features onto a dataframe
    df = demo.merge(ri_util, on="subject_id", how="left")
    df = df.merge(rx, on="subject_id", how="left")
    df = df.merge(y_oud, on="subject_id", how="left")
    df = df.merge(y_eligibility, on="subject_id", how="left")
    df = df.merge(bmi_features, on="subject_id", how="left")
    df = df.merge(drg_features, on="subject_id", how="left")
    df = df.merge(icu_features, on="subject_id", how="left")

    # Fill missing values
    # strings
    df["race"] = df["race"].fillna("UNKNOWN")
    df["insurance"] = df["insurance"].fillna("UNKNOWN")

    # numerics
    cols_to_fill = [
        "n_hospital_admits", "avg_los_days", "total_los_days", "opioid_rx_count",
        "opioid_hadms", "distinct_opioids", "opioid_exposure_days", "any_benzo_flag",
        "any_opioid_flag", "bmi", "has_bmi", "obesity_flag", "morbid_obesity_flag",
        "avg_drg_severity", "max_drg_severity", "avg_drg_mortality", "max_drg_mortality",
        "high_severity_flag", "high_mortality_flag", "n_admissions_with_drg",
        "n_icu_stays", "total_icu_hours", "total_icu_days", "n_icu_admissions", "has_icu_stay"
    ]

    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # targets TODO to need to remove will_get_opiod_rx
    df["y_oud"] = df["y_oud"].fillna(0).astype(int)
    df["will_get_opioid_rx"] = df["any_opioid_flag"].fillna(0).astype(int)
    df["opioid_eligibility"] = df["opioid_eligibility"].fillna(0).astype(int)

    print(f"Total patients: {len(df):,}")
    print(f"Total features (before SHP): {len(df.columns) - 3}")  # Remove targets


    print(f"OUD positive cases: {df['y_oud'].sum():,} ({df['y_oud'].mean()*100:.1f}%)")
    print(f"Opioid eligible (has pain): {df['opioid_eligibility'].sum():,} ({df['opioid_eligibility'].mean()*100:.1f}%)")
    print(f"Received opioid prescriptions: {df['any_opioid_flag'].sum():,} ({df['any_opioid_flag'].mean()*100:.1f}%)")

    # We are splitting total data into multiple buckets
    # 30% for main model testing
    # 56% for main model training
    # 7% for main model validation
    # 7% for SHAP feature selection (only this 7% used in this file)

    # Test data
    df_dev, df_test = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['y_oud']
    )


    # 7% for SHAP feature selection
    df_shap, df_remaining = train_test_split(
        df_dev, test_size=0.90, random_state=42, stratify=df_dev['y_oud']
    )

    # train and validation split    
    df_train, df_val = train_test_split(
        df_remaining, test_size=0.10, random_state=42, stratify=df_remaining['y_oud']
    )
  
    # Save these in temp dir
    df_shap.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_subset.csv"), index=False)

    # Now real SHAP code
    
    X, y_oud, y_rx, y_eligibility, feature_cols, label_encoders = prepare_features_for_shap(df_shap)

    # For OUD
    oud_features_all, oud_importance = get_shap_feature_importance(
        X, y_oud, "y_oud (OUD Risk - Preventive Model)", top_n=25
    )
    
    # For Eligibility exclude opiod exposure related features. They have are too related to eligibility

    opioid_features = ['opioid_rx_count', 'opioid_hadms', 'distinct_opioids', 
                       'opioid_exposure_days', 'any_opioid_flag']
    X_eligibility = X[[col for col in X.columns if col not in opioid_features]]
    eligibility_feature_cols = [col for col in feature_cols if col not in opioid_features]
    
    eligibility_features_all, eligibility_importance = get_shap_feature_importance(
        X_eligibility, y_eligibility, "opioid_eligibility (Eligibility - Clinical Need)", top_n=25
    )

    # For checking to see the right percentile, where real importantnce drops off
    percentiles = [50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(oud_importance['shap_importance'], p)
        count = (oud_importance['shap_importance'] >= val).sum()
        print(f"      {p}th percentile: {val:.6f} ({count} features)")
    
    print("Eligibility Model - Feature importance percentiles:")
    for p in percentiles:
        val = np.percentile(eligibility_importance['shap_importance'], p)
        count = (eligibility_importance['shap_importance'] >= val).sum()
        print(f"      {p}th percentile: {val:.6f} ({count} features)")
    
    # Select features above 50th percentile for each target (decided after looking at data)
    oud_threshold = np.percentile(oud_importance['shap_importance'], 50)
    eligibility_threshold = np.percentile(eligibility_importance['shap_importance'], 50)
    
    oud_features = oud_importance[oud_importance['shap_importance'] >= oud_threshold]['feature'].tolist()
    eligibility_features = eligibility_importance[eligibility_importance['shap_importance'] >= eligibility_threshold]['feature'].tolist()
    
    print(f"OUD Risk Model: {len(oud_features)} features selected (threshold: {oud_threshold:.6f})")
    print(f"Eligibility Model: {len(eligibility_features)} features selected (threshold: {eligibility_threshold:.6f})")

    # Combine important features from both models
    all_important_features = list(set(oud_features + eligibility_features))
    print(f"Combined unique important features: {len(all_important_features)}")


    # Save SHAP importance results
    importance_df = pd.DataFrame({
        "feature": all_important_features,
        "selected_for_oud": [f in oud_features for f in all_important_features],
        "selected_for_eligibility": [f in eligibility_features for f in all_important_features],
    })
    importance_df.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_feature_importance.csv"), index=False)
    
    # Save detailed importance scores
    oud_importance.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_importance_oud.csv"), index=False)
    eligibility_importance.to_csv(os.path.join(INTERMEDIATE_DIR, "shap_importance_eligibility.csv"), index=False)
    print(f"  ✓ Saved feature importance results to temp_data/")

    # Prepare final datasets with selected features

    final_features = ["subject_id"] + all_important_features + ["y_oud", "opioid_eligibility"]
    
    # Training set (with selected features only)
    df_train_processed = df_train[final_features].copy()
    # Fill NaN values with 0 
    df_train_processed = df_train_processed.fillna(0)   
    df_train_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_data.csv"), index=False)
    
    # Validation set (with selected features only)
    df_val_processed = df_val[final_features].copy()
    # Fill NaN values with 0
    df_val_processed = df_val_processed.fillna(0)
    df_val_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "validation_data.csv"), index=False)
    
    # Test set (with selected features only)
    df_test_processed = df_test[final_features].copy()
    # Fill NaN values with 0
    df_test_processed = df_test_processed.fillna(0)   
    df_test_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_data.csv"), index=False)
    
    # Full dataset with selected features (for reference)
    df_full_processed = df[final_features].copy()
    # Fill NaN values with 0
    numeric_cols = df_full_processed.select_dtypes(include=[np.number]).columns
    df_full_processed[numeric_cols] = df_full_processed[numeric_cols].fillna(0)
    df_full_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "full_data_selected_features.csv"), index=False)

    # Save stats
    metadata = {
        "description": "Two-model opioid audit system: Eligibility + OUD Risk",
        "audit_logic": "Flag if (Eligibility=NO) OR (OUD_Risk=HIGH)",
        "total_records": len(df),
        "train_records": len(df_train),
        "validation_records": len(df_val),
        "test_records": len(df_test),
        "shap_records": len(df_shap),
        "original_features": len(feature_cols),
        "selected_features": len(all_important_features),
        "feature_names": all_important_features,
        "oud_features": oud_features,
        "eligibility_features": eligibility_features,
        "oud_positive_train": int(df_train['y_oud'].sum()),
        "oud_positive_validation": int(df_val['y_oud'].sum()),
        "oud_positive_test": int(df_test['y_oud'].sum()),
        "eligibility_positive_train": int(df_train['opioid_eligibility'].sum()),
        "eligibility_positive_validation": int(df_val['opioid_eligibility'].sum()),
        "eligibility_positive_test": int(df_test['opioid_eligibility'].sum()),
        "data_split_strategy": "7% SHAP, 56% train, 7% validation, 30% test"
    }
    
    import json
    with open(os.path.join(PROCESSED_DATA_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
