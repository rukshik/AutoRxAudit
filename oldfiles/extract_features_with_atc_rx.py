import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

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

def compute_age_at_first_admit(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    first_admit = admissions.sort_values("admittime").groupby("subject_id", as_index=False).first()[["subject_id", "hadm_id", "admittime"]]
    demo = patients.merge(first_admit, on="subject_id", how="left")
    demo["admittime"] = pd.to_datetime(demo["admittime"], errors="coerce")
    admit_year = demo["admittime"].dt.year
    demo["age_at_first_admit"] = admit_year - (demo["anchor_year"] - demo["anchor_age"])
    demo["age_at_first_admit"] = demo["age_at_first_admit"].clip(lower=0, upper=120).fillna(0)
    return demo[["subject_id", "gender", "age_at_first_admit"]]

def aggregate_race_insurance(admissions: pd.DataFrame) -> pd.DataFrame:
    agg = admissions.groupby("subject_id").agg(
        race=("race", mode_or_nan),
        insurance=("insurance", mode_or_nan),
        n_hospital_admits=("hadm_id", "nunique"),
        avg_los_days=("los", "mean"),
        total_los_days=("los", "sum"),
    ).reset_index()
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
    "codeine", "tramadol", "oxymorphone", "tapentadol", "methadone", "buprenorphine"
]

BENZO_PATTERNS = [
    "diazepam", "lorazepam", "alprazolam", "clonazepam", "temazepam",
    "chlordiazepoxide", "midazolam", "oxazepam"
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
    rx["is_opioid"] = rx["drug"].apply(lambda s: any(p in str(s).lower() for p in OPIOID_PATTERNS))
    rx["is_benzo"]  = rx["drug"].apply(lambda s: any(p in str(s).lower() for p in BENZO_PATTERNS))
    opioid = rx[rx["is_opioid"]].copy()
    benzo  = rx[rx["is_benzo"]].copy()
    other  = rx[(~rx["is_opioid"]) & (~rx["is_benzo"])]
    # ATC grouping for "other"
    other["atc_group"] = other["drug"].apply(atc_group)
    atc_counts = other.groupby(["subject_id", "atc_group"]).size().unstack(fill_value=0)
    atc_counts.columns = [f"atc_{col}_rx_count" for col in atc_counts.columns]
    atc_counts = atc_counts.reset_index()
    # Aggregate as before
    def agg_rx(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["subject_id",
                                         f"{kind}_rx_count",
                                         f"{kind}_hadms",
                                         f"distinct_{kind}s",
                                         f"{kind}_exposure_days"])
        df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
        df["stoptime"]  = pd.to_datetime(df["stoptime"], errors="coerce")
        dur = (df["stoptime"] - df["starttime"]).dt.total_seconds() / 86400.0
        df["duration_days"] = np.maximum(0.0, np.nan_to_num(dur))
        g = df.groupby("subject_id").agg(
            **{
                f"{kind}_rx_count": ("drug", "count"),
                f"{kind}_hadms":    ("hadm_id", "nunique"),
                f"distinct_{kind}s":("drug", "nunique"),
                f"{kind}_exposure_days": ("duration_days", "sum"),
            }
        ).reset_index()
        return g
    opioid_agg = agg_rx(opioid, "opioid")
    benzo_agg = agg_rx(benzo, "benzo")
    # Flags
    benzo_flag = benzo.groupby("subject_id").size().reset_index(name="cnt") if not benzo.empty else pd.DataFrame(columns=["subject_id", "cnt"])
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
    # Fill missing ATC columns with 0
    for col in atc_counts.columns:
        if col != "subject_id":
            out[col] = out[col].fillna(0)
    return out

def build_oud_label(diagnoses: pd.DataFrame) -> pd.DataFrame:
    d = diagnoses.copy()
    if "icd_version" not in d.columns or "icd_code" not in d.columns:
        raise ValueError("Expected columns 'icd_version' and 'icd_code' in diagnoses_icd.csv.gz")
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
    lbl = d.groupby("subject_id")["is_oud"].max().reset_index().rename(columns={"is_oud":"y_oud"})
    return lbl

def main():
    DEMO_DIR = os.path.join("data", "mimic-clinical-iv-demo", "hosp")
    patients = read_csv("patients.csv.gz", DEMO_DIR)
    admissions_raw = read_csv("admissions.csv.gz", DEMO_DIR)
    diagnoses = read_csv("diagnoses_icd.csv.gz", DEMO_DIR)
    prescriptions = read_csv("prescriptions.csv.gz", DEMO_DIR)
    admissions = compute_los_days(admissions_raw)
    ri_util = aggregate_race_insurance(admissions)
    demo = compute_age_at_first_admit(patients, admissions)
    rx = build_rx_features_atc(prescriptions)
    y = build_oud_label(diagnoses)
    df = demo.merge(ri_util, on="subject_id", how="left") \
             .merge(rx, on="subject_id", how="left") \
             .merge(y, on="subject_id", how="left")
    df["race"] = df["race"].fillna("UNKNOWN")
    df["insurance"] = df["insurance"].fillna("UNKNOWN")
    for col in ["n_hospital_admits", "avg_los_days", "total_los_days",
                "opioid_rx_count", "opioid_hadms", "distinct_opioids",
                "opioid_exposure_days", "any_benzo_flag", "any_opioid_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["y_oud"] = df["y_oud"].fillna(0).astype(int)
    df["will_get_opioid_rx"] = df["any_opioid_flag"].fillna(0).astype(int)
    out_csv = "mimiciv_demo_oud_features_with_atc_rx.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote features: {out_csv}")
    print(df.head())

if __name__ == "__main__":
    main()
