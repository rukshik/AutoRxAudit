import pandas as pd
import os

# Check training data
print("=" * 80)
print("Training Data Analysis (10000_v3)")
print("=" * 80)

train = pd.read_csv('ai-layer/processed_data/10000_v3/train_data.csv')
test = pd.read_csv('ai-layer/processed_data/10000_v3/test_data.csv')
val = pd.read_csv('ai-layer/processed_data/10000_v3/validation_data.csv')

all_patients = pd.concat([train, test, val])['subject_id'].unique()
total_patients = len(all_patients)

print(f"\nTotal unique patients: {total_patients}")
print(f"  Train: {train['subject_id'].nunique()}")
print(f"  Test: {test['subject_id'].nunique()}")  
print(f"  Val: {val['subject_id'].nunique()}")

# Check if patient 20000199 is in the data
print(f"\nPatient 20000199 in training data: {20000199 in all_patients}")

# Check available datasets
print("\n" + "=" * 80)
print("Available Dataset Folders")
print("=" * 80)

datasets_dir = 'datasets'
if os.path.exists(datasets_dir):
    for item in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, item)
        if os.path.isdir(path):
            print(f"  {item}")

# Check what data was loaded
print("\n" + "=" * 80)
print("Current DB Data Source")
print("=" * 80)
print("Loaded from: data/mimic-clinical-iv-demo/hosp/prescriptions.csv.gz")
print("This is the DEMO dataset (100 patients)")

# Check if 10000 dataset exists
print("\n" + "=" * 80)
print("Checking for 10000-patient dataset")
print("=" * 80)

possible_paths = [
    'datasets/synthetic_mimic_10000_v3/mimic-clinical-iv-demo/hosp/prescriptions.csv.gz',
    'datasets/synthetic_mimic_1000_v3/mimic-clinical-iv-demo/hosp/prescriptions.csv.gz',
]

for path in possible_paths:
    if os.path.exists(path):
        import gzip
        df = pd.read_csv(path, compression='gzip', nrows=1)
        with gzip.open(path, 'rt') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"✓ Found: {path}")
        print(f"  Has data (checked 1 row)")
