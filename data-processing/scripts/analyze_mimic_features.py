"""
Analyze MIMIC-IV demo data to find features that could improve eligibility prediction
"""
import pandas as pd
import os

data_dir = "data/mimic-clinical-iv-demo/hosp"

print("=" * 80)
print("MIMIC-IV DEMO DATA FEATURE ANALYSIS")
print("=" * 80)

# Files we're currently using
current_files = ["patients.csv.gz", "admissions.csv.gz", "diagnoses_icd.csv.gz", "prescriptions.csv.gz"]

# All available files
all_files = [
    "procedures_icd.csv.gz",  # Surgical procedures
    "labevents.csv.gz",       # Lab test results
    "omr.csv.gz",             # Organ-specific measurements
    "services.csv.gz",        # Clinical service transfers
    "drgcodes.csv.gz",        # DRG codes (diagnosis-related groups)
    "transfers.csv.gz",       # ICU/ward transfers
    "emar.csv.gz",            # Electronic medication administration
    "poe.csv.gz",             # Provider order entry
]

print("\n📊 ANALYZING AVAILABLE DATA FILES...\n")

for file in all_files:
    file_path = os.path.join(data_dir, file)
    if not os.path.exists(file_path):
        print(f"❌ {file}: NOT FOUND")
        continue
    
    try:
        # Read first few rows to get structure
        df = pd.read_csv(file_path, nrows=100)
        print(f"✓ {file}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape[0]} rows (sample)")
        print(f"  Key info: {df.dtypes.value_counts().to_dict()}")
        
        # Special analysis for potentially useful files
        if file == "procedures_icd.csv.gz":
            print(f"  → Surgical procedures could indicate:")
            print(f"     - Back surgery → chronic pain")
            print(f"     - Joint replacement → orthopedic pain")
            print(f"     - Cancer surgery → oncologic pain")
            if 'icd_code' in df.columns:
                print(f"  → Sample codes: {df['icd_code'].head(5).tolist()}")
        
        elif file == "labevents.csv.gz":
            print(f"  → Lab tests could indicate:")
            print(f"     - Inflammatory markers (CRP, ESR) → pain/inflammation")
            print(f"     - Hemoglobin → anemia/pain")
            if 'itemid' in df.columns:
                print(f"  → Unique tests: {df['itemid'].nunique()}")
        
        elif file == "services.csv.gz":
            print(f"  → Clinical services could indicate:")
            print(f"     - Pain service transfer → documented pain management")
            print(f"     - Oncology → cancer-related pain")
            if 'curr_service' in df.columns:
                print(f"  → Services: {df['curr_service'].value_counts().head(5).to_dict()}")
        
        elif file == "omr.csv.gz":
            print(f"  → OMR (organ measurements) could include:")
            print(f"     - Pain scores")
            print(f"     - BMI (obesity → back pain)")
            if 'result_name' in df.columns:
                print(f"  → Measurements: {df['result_name'].value_counts().head(10).to_dict()}")
        
        print()
    
    except Exception as e:
        print(f"❌ {file}: ERROR - {e}\n")

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR ELIGIBILITY MODEL")
print("=" * 80)

print("""
🎯 TOP FEATURES TO ADD:

1. **procedures_icd.csv.gz** - HIGHEST PRIORITY
   - Surgical procedures are strong indicators of pain
   - Back surgery (lumbar fusion, laminectomy) → chronic pain
   - Joint procedures (knee/hip replacement) → orthopedic pain
   - Cancer surgery → oncologic pain
   - Feature: procedure_count, pain_related_procedure_flag

2. **services.csv.gz** - HIGH PRIORITY
   - Clinical service transfers indicate diagnosis severity
   - Pain management service → documented pain
   - Oncology service → cancer pain
   - Feature: pain_service_flag, oncology_service_flag

3. **omr.csv.gz** - MEDIUM PRIORITY
   - May contain pain scores directly
   - BMI data → obesity-related pain
   - Feature: has_pain_score, high_bmi_flag

4. **labevents.csv.gz** - LOW PRIORITY (but useful)
   - Inflammatory markers (CRP, ESR) → inflammation/pain
   - Feature: elevated_inflammation_markers

IMPLEMENTATION:
- Add to feature_engineering in shap_feature_selection.py
- These features DON'T use opioid history (no data leakage)
- Could improve Eligibility AUC from 64% → 70-75%
""")
