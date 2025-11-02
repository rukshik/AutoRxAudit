"""
Quick analysis: Check if MIMIC demo data has pain-related diagnoses
that can be used for opioid eligibility modeling
"""

import pandas as pd

# Load data
print("Loading data...")
dx = pd.read_csv('data/mimic-clinical-iv-demo/hosp/diagnoses_icd.csv.gz')
d_icd = pd.read_csv('data/mimic-clinical-iv-demo/hosp/d_icd_diagnoses.csv.gz')

# Merge to get descriptions
dx_with_desc = dx.merge(d_icd, on=['icd_code', 'icd_version'], how='left')

# Identify pain-related conditions
pain_keywords = [
    'pain', 'trauma', 'injury', 'fracture', 'cancer', 'neoplasm',
    'postoperative', 'surgery', 'operation', 'wound', 'burn'
]

pattern = '|'.join(pain_keywords)
dx_with_desc['is_pain_related'] = dx_with_desc['long_title'].str.contains(
    pattern, case=False, na=False
)

# Calculate statistics
pain_patients = dx_with_desc[dx_with_desc['is_pain_related']==True]['subject_id'].nunique()
total_patients = dx['subject_id'].nunique()
pain_records = dx_with_desc['is_pain_related'].sum()

print(f"\n{'='*60}")
print("PAIN-RELATED DIAGNOSIS ANALYSIS")
print(f"{'='*60}")
print(f"Total patients in demo: {total_patients}")
print(f"Patients with pain-related diagnoses: {pain_patients} ({pain_patients/total_patients*100:.1f}%)")
print(f"Total pain-related diagnosis records: {pain_records}")

# Show some examples
print(f"\n{'='*60}")
print("Sample pain-related diagnoses:")
print(f"{'='*60}")
sample = dx_with_desc[dx_with_desc['is_pain_related']==True][
    ['subject_id', 'icd_code', 'icd_version', 'long_title']
].head(15)
print(sample.to_string(index=False))

print(f"\n{'='*60}")
print("CONCLUSION:")
print(f"{'='*60}")
if pain_patients > 0:
    print("✅ YES! We can use MIMIC demo data to predict opioid eligibility")
    print(f"   {pain_patients} patients have pain-related conditions")
    print("   These can be used as indicators for legitimate opioid need")
else:
    print("❌ No pain-related diagnoses found in demo data")

