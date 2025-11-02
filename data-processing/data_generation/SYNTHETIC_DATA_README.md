# Synthetic Data Generation & Proper Data Splitting Guide

## Overview

This guide explains the complete workflow from generating synthetic data (50,000 records from 100 samples) to properly splitting data for feature selection and Neural Network training.

## 📁 Folder Structure

```
AutoRxAudit/
├── data/                                    # Original 100-record sample data
│   └── mimic-clinical-iv-demo/hosp/
│       ├── patients.csv.gz
│       ├── admissions.csv.gz
│       ├── diagnoses_icd.csv.gz
│       └── prescriptions.csv.gz
│
├── data_generation/                         # Synthetic data generation scripts
│   ├── generate_synthetic_data.py          # Main generator
│   ├── run_generation.py                   # Runner script
│   └── README.md                           # Documentation
│
├── synthetic_data/                          # Generated 50K records
│   └── mimic-clinical-iv-demo/hosp/
│       ├── patients.csv.gz                 # 50,000 patients
│       ├── admissions.csv.gz               # ~200K admissions
│       ├── diagnoses_icd.csv.gz            # ~800K diagnoses
│       └── prescriptions.csv.gz            # ~1.6M prescriptions
│
├── extract_features_with_shap.py           # Original SHAP script (for 100 records)
├── extract_features_with_shap_synthetic.py # SHAP script for synthetic data (needs shap package)
├── data_splitting_demo.py                  # Proper data splitting (works without shap)
│
├── neural_network_training_data.csv        # 31,500 records for your NN (63%)
├── neural_network_final_test_data.csv      # 15,000 records for final test (30%)
├── feature_importance_results.csv          # Feature selection details
└── full_dataset_selected_features.csv      # Complete 50K with selected features
```

## 🔄 Complete Workflow

### Step 1: Generate Synthetic Data (DONE ✓)

```bash
cd data_generation
python run_generation.py
```

**Result:** 50,000 synthetic patient records created in `synthetic_data/` folder

### Step 2: Feature Selection & Data Splitting (DONE ✓)

```bash
python data_splitting_demo.py
```

**This script does THREE critical things:**

1. **Loads synthetic data** (50,000 records)
2. **Implements proper data splitting:**
   - 7% (3,500 records) → Feature selection
   - 63% (31,500 records) → Neural Network training
   - 30% (15,000 records) → Final holdout test
3. **Selects important features** using Random Forest importance

## 📊 Data Splitting Strategy

### Why This Matters

Your original question: "Does SHAP use all my data?"
- **Answer:** YES, it was using ALL data (bad!)
- **Problem:** No data left for Neural Network training
- **Solution:** We implemented proper data splitting

### The Split Breakdown (50,000 Records Total)

```
┌─────────────────────────────────────────────────────┐
│            50,000 Total Records                     │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
    ┌───▼────────────────┐      ┌──────▼──────┐
    │  Development Pool  │      │ Final Test  │
    │   35,000 (70%)     │      │ 15,000 (30%)│
    │                    │      │             │
    │  ┌──────┬────────┐ │      │  HOLD OUT   │
    │  │ SHAP │  NN    │ │      │  Until      │
    │  │3,500 │31,500  │ │      │  Final      │
    │  │ (7%) │ (63%)  │ │      │  Evaluation │
    │  └──────┴────────┘ │      │             │
    └────────────────────┘      └─────────────┘
```

### Inside SHAP Subset (3,500 records)

The `data_splitting_demo.py` gives these 3,500 records to the feature importance function, which then internally splits them:

```
3,500 records for feature selection
    │
    ├─ 2,450 (70%) → Train Random Forest
    └─ 1,050 (30%) → Calculate importance on test set
```

**This is perfect because:**
- ✅ Feature importance calculated on unseen data (no overfitting)
- ✅ Sample size (1,050) is sufficient for stable importance scores
- ✅ Fast computation
- ✅ No data leakage to Neural Network training

## 📝 Output Files Explained

### For Your Neural Network

1. **`neural_network_training_data.csv`** (31,500 records)
   - Use this for training your Neural Network
   - Contains 17 selected features
   - Split this further into train/validation (e.g., 80/20)
   - OUD cases: ~5,644 (17.9%)

2. **`neural_network_final_test_data.csv`** (15,000 records)
   - **DO NOT touch during development!**
   - Use ONLY for final model evaluation
   - Same 17 features as training data
   - OUD cases: ~2,688 (17.9%)

3. **`feature_importance_results.csv`**
   - Shows which features were selected
   - Indicates if feature was important for OUD or opioid prescription prediction

### For Reference

4. **`full_dataset_selected_features.csv`** (50,000 records)
   - Complete dataset with only the 17 selected features
   - Useful for exploratory data analysis
   - Don't use for training (use split datasets above)

## 🎯 Selected Features (17 Total)

The feature selection identified these as most important:

1. `age_at_first_admit` - Patient age at first admission
2. `atc_Other_rx_count` - Other medication prescriptions
3. `opioid_hadms` - Number of admissions with opioid prescriptions
4. `atc_N_rx_count` - Nervous system medications (antidepressants, etc.)
5. `atc_A_rx_count` - Digestive/metabolism medications
6. `opioid_exposure_days` - Total days of opioid exposure
7. `atc_C_rx_count` - Cardiovascular medications
8. `atc_J_rx_count` - Anti-infective medications
9. `n_hospital_admits` - Total number of hospital admissions
10. `atc_H_rx_count` - Hormonal medications
11. `atc_B_rx_count` - Blood-related medications
12. `avg_los_days` - Average length of stay
13. `any_opioid_flag` - Binary: received any opioid
14. `opioid_rx_count` - Total opioid prescriptions
15. `total_los_days` - Total length of stay across admissions
16. `atc_R_rx_count` - Respiratory medications
17. `distinct_opioids` - Number of different opioids prescribed

## 🧠 Neural Network Training Guide

### Recommended Approach

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your training data
df_train = pd.read_csv('neural_network_training_data.csv')

# Separate features and targets
X = df_train.drop(['subject_id', 'y_oud', 'will_get_opioid_rx'], axis=1)
y_oud = df_train['y_oud']
y_rx = df_train['will_get_opioid_rx']

# Further split into train/validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_oud,  # or y_rx depending on your target
    test_size=0.20,
    random_state=42,
    stratify=y_oud
)

# Now train your Neural Network
# X_train: 25,200 records
# X_val: 6,300 records

# After all development is done, load test set for final evaluation
df_test = pd.read_csv('neural_network_final_test_data.csv')
X_test = df_test.drop(['subject_id', 'y_oud', 'will_get_opioid_rx'], axis=1)
y_test = df_test['y_oud']  # or y_rx
```

### Final Data Distribution

```
Total 50,000 records:
├── Feature Selection: 3,500 (7%) ← Used only for feature selection
├── NN Training: 25,200 (50.4%) ← Train your model
├── NN Validation: 6,300 (12.6%) ← Tune hyperparameters
└── Final Test: 15,000 (30%) ← Final evaluation only
```

## ❓ FAQ

### Q: Why not use all 50K for Neural Network training?
**A:** Feature selection on the same data you train on causes overfitting. The features might work great on training data but fail on new data.

### Q: Is 3,500 records enough for feature selection?
**A:** Yes! Random Forest can identify important patterns with even 1,000 records. The feature importance generalizes well.

### Q: Should I use SHAP or Random Forest importance?
**A:** For your use case, both are fine:
- **Random Forest importance**: Fast, already implemented, works well
- **SHAP**: More theoretically sound, but requires `shap` package installation
- The selected features will be very similar

### Q: Can I change the split proportions?
**A:** Yes! In `data_splitting_demo.py`, modify:
```python
# Change test_size values to adjust proportions
df_dev, df_final_test = train_test_split(df, test_size=0.30, ...)  # 30% test
df_feature_selection, df_neural_network = train_test_split(df_dev, test_size=0.90, ...)  # 10% features
```

### Q: What about the original 100-record data?
**A:** Keep it! It's in the `data/` folder and remains untouched. The synthetic data is separate in `synthetic_data/`.

## ✅ Checklist Before Training Your Neural Network

- [x] Synthetic data generated (50,000 records)
- [x] Data properly split (no leakage)
- [x] Features selected (17 features)
- [x] Training data saved (`neural_network_training_data.csv`)
- [x] Test data saved and held out (`neural_network_final_test_data.csv`)
- [ ] Split training data into train/validation
- [ ] Build Neural Network architecture
- [ ] Train and tune model on training/validation data
- [ ] Final evaluation on test data

## 🎉 You're Ready!

You now have:
- ✅ 50,000 high-quality synthetic medical records
- ✅ Properly split data with no leakage
- ✅ 17 carefully selected features
- ✅ 31,500 records for Neural Network training
- ✅ 15,000 records for final testing
- ✅ Clear understanding of the data pipeline

**Next step:** Build and train your Neural Network using `neural_network_training_data.csv`!