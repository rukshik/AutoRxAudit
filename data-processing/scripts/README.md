# Data Processing Scripts

This folder contains utility scripts for data validation, analysis, and checking.

## Scripts

### Data Validation & Checking
- **check_synthetic_data.py** - Validates synthetic MIMIC-IV data structure and content
- **check_pain_diagnoses.py** - Analyzes pain diagnosis codes in the dataset
- **check_oud_signals.py** - Checks for OUD (Opioid Use Disorder) signals in data
- **check_results.py** - Validates processing results and data integrity

### Data Analysis
- **analyze_mimic_features.py** - Basic feature analysis of MIMIC-IV data
- **analyze_mimic_features_detailed.py** - Detailed statistical analysis of features
- **verify_new_features.py** - Verifies newly added features (BMI, DRG, ICU)

### Demo Scripts
- **data_splitting_demo.py** - Demonstrates train/validation/test splitting strategies

## Usage

These scripts are primarily used for:
1. **Data Quality Assurance** - Validating synthetic data generation
2. **Feature Engineering Validation** - Verifying new features are correctly generated
3. **Exploratory Data Analysis** - Understanding dataset characteristics
4. **Debugging** - Troubleshooting data processing issues

## Note

These are utility/analysis scripts and are not part of the production pipeline. The main data processing pipeline consists of:
1. `data_generation/generate_synthetic_data.py` - Generate synthetic MIMIC-IV data
2. `data_generation/extract_pain_codes.py` - Extract pain-related ICD codes
3. `../feature_selection/shap_feature_selection.py` - Feature selection using SHAP
