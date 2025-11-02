import pandas as pd
import numpy as np
import sys
import os

# Get dataset size from command line argument (default: 1000)
if len(sys.argv) > 1:
    dataset_size = sys.argv[1]
else:
    dataset_size = '1000'

# Check if processed data exists
data_path = f'../processed_data/{dataset_size}/train_data.csv'
if not os.path.exists(data_path):
    print(f"Error: {data_path} does not exist!")
    print(f"Please run feature selection first:")
    print(f"  python shap_feature_selection.py --input-dir ../../synthetic_data/{dataset_size}/mimic-iv-synthetic --output-dir ../processed_data/{dataset_size} --temp-dir temp_data_{dataset_size}")
    sys.exit(1)

print(f"Checking dataset: {dataset_size} patients")
print(f"Data path: {data_path}\n")

# Load training data
train = pd.read_csv(data_path)

# Split by OUD status
oud = train[train['y_oud'] == 1]
non_oud = train[train['y_oud'] == 0]

print(f"=== OUD Signal Analysis ===")
print(f"OUD patients: {len(oud)} ({len(oud)/len(train)*100:.1f}%)")
print(f"Non-OUD patients: {len(non_oud)} ({len(non_oud)/len(train)*100:.1f}%)")
print()

# Get all numeric features from the data (excluding IDs and targets)
exclude_cols = ['subject_id', 'y_oud', 'opioid_eligibility', 'insurance']
all_features = [col for col in train.columns if col not in exclude_cols]

# Prioritize opioid-specific features if they exist
priority_features = ['opioid_rx_count', 'distinct_opioids', 'opioid_exposure_days', 'opioid_hadms']
features = [f for f in priority_features if f in all_features]

# Add other numeric features
other_features = [f for f in all_features if f not in features and train[f].dtype in ['int64', 'float64']]
features.extend(other_features)

print(f"Available features ({len(features)}): {features}\n")

print("=== Feature Comparison: OUD vs Non-OUD ===\n")
for feature in features:
    oud_mean = oud[feature].mean()
    non_oud_mean = non_oud[feature].mean()
    
    if non_oud_mean > 0:
        ratio = oud_mean / non_oud_mean
    else:
        ratio = float('inf') if oud_mean > 0 else 1.0
    
    print(f"{feature}:")
    print(f"  OUD:     {oud_mean:.2f}")
    print(f"  Non-OUD: {non_oud_mean:.2f}")
    print(f"  Ratio:   {ratio:.2f}x")
    print()

# Check OPIOID prescriptions if available (most important signal!)
if 'opioid_rx_count' in train.columns:
    print("=== OPIOID Prescription Distribution ===")
    print(f"\nOUD patients with opioids: {(oud['opioid_rx_count'] > 0).sum()} / {len(oud)} ({(oud['opioid_rx_count'] > 0).sum()/len(oud)*100:.1f}%)")
    print(f"Non-OUD patients with opioids: {(non_oud['opioid_rx_count'] > 0).sum()} / {len(non_oud)} ({(non_oud['opioid_rx_count'] > 0).sum()/len(non_oud)*100:.1f}%)")

    print(f"\nOUD patients - Opioid Rx count distribution:")
    print(f"  Min: {oud['opioid_rx_count'].min()}")
    print(f"  25%: {oud['opioid_rx_count'].quantile(0.25)}")
    print(f"  50%: {oud['opioid_rx_count'].median()}")
    print(f"  75%: {oud['opioid_rx_count'].quantile(0.75)}")
    print(f"  Max: {oud['opioid_rx_count'].max()}")

    print(f"\nNon-OUD patients - Opioid Rx count distribution:")
    print(f"  Min: {non_oud['opioid_rx_count'].min()}")
    print(f"  25%: {non_oud['opioid_rx_count'].quantile(0.25)}")
    print(f"  50%: {non_oud['opioid_rx_count'].median()}")
    print(f"  75%: {non_oud['opioid_rx_count'].quantile(0.75)}")
    print(f"  Max: {non_oud['opioid_rx_count'].max()}")
else:
    print("=== NOTE: Opioid-specific features not selected by SHAP ===")
    print("Using proxy features (atc_N_rx_count includes opioids)")


# Check eligibility vs OUD
print("\n=== Eligibility vs OUD ===")
print(f"OUD patients marked as eligible: {(oud['opioid_eligibility'] == 1).sum()} / {len(oud)} ({(oud['opioid_eligibility'] == 1).sum()/len(oud)*100:.1f}%)")
print(f"Non-OUD patients marked as eligible: {(non_oud['opioid_eligibility'] == 1).sum()} / {len(non_oud)} ({(non_oud['opioid_eligibility'] == 1).sum()/len(non_oud)*100:.1f}%)")
