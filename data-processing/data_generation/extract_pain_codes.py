"""
Extract pain-related ICD codes from MIMIC-IV demo data
to use as ground truth for opioid eligibility in synthetic data
"""

import pandas as pd
import json

# Load ICD dictionary
print("Loading ICD diagnosis dictionary...")
d_icd = pd.read_csv('../../data/mimic-clinical-iv-demo/hosp/d_icd_diagnoses.csv.gz')

# Define pain-related keywords (based on actual MIMIC data analysis)
pain_keywords = [
    'pain', 'trauma', 'injury', 'fracture', 'cancer', 'neoplasm',
    'postoperative', 'surgery', 'operation', 'wound', 'burn',
    'malignant', 'carcinoma', 'sarcoma', 'tumor',
    'accident', 'hemorrhage', 'laceration', 'contusion'
]

# Find pain-related codes
pattern = '|'.join(pain_keywords)
pain_codes = d_icd[d_icd['long_title'].str.contains(pattern, case=False, na=False)].copy()

print(f"\nFound {len(pain_codes)} pain-related ICD codes")
print(f"ICD-9: {len(pain_codes[pain_codes['icd_version']==9])} codes")
print(f"ICD-10: {len(pain_codes[pain_codes['icd_version']==10])} codes")

# Save as JSON for easy integration
pain_codes_dict = {
    'icd9': pain_codes[pain_codes['icd_version']==9]['icd_code'].tolist(),
    'icd10': pain_codes[pain_codes['icd_version']==10]['icd_code'].tolist(),
    'description': 'ICD codes indicating legitimate pain/medical need for opioids',
    'keywords': pain_keywords
}

output_file = 'pain_diagnosis_codes.json'
with open(output_file, 'w') as f:
    json.dump(pain_codes_dict, f, indent=2)

print(f"\n✅ Saved {len(pain_codes_dict['icd9']) + len(pain_codes_dict['icd10'])} pain codes to {output_file}")

# Show some examples
print("\nExample ICD-9 pain codes:")
sample_9 = pain_codes[pain_codes['icd_version']==9][['icd_code', 'long_title']].head(10)
for idx, row in sample_9.iterrows():
    print(f"  {row['icd_code']}: {row['long_title'][:70]}...")

print("\nExample ICD-10 pain codes:")
sample_10 = pain_codes[pain_codes['icd_version']==10][['icd_code', 'long_title']].head(10)
for idx, row in sample_10.iterrows():
    print(f"  {row['icd_code']}: {row['long_title'][:70]}...")
