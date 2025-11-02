# Synthetic Data Generation

This folder contains scripts to expand the original 100-record MIMIC-IV demo dataset to 50,000 synthetic records.

## Files

- `generate_synthetic_data.py` - Main synthetic data generator class
- `run_generation.py` - Simple runner script

## Usage

1. Navigate to this folder:
```bash
cd data_generation
```

2. Run the generation:
```bash
python run_generation.py
```

## Output

The synthetic data will be created in the `../synthetic_data/mimic-clinical-iv-demo/hosp/` folder with these files:

- `patients.csv.gz` - 50,000 synthetic patients
- `admissions.csv.gz` - Hospital admissions (1-8 per patient)
- `diagnoses_icd.csv.gz` - ICD diagnosis codes (1-8 per admission)
- `prescriptions.csv.gz` - Medication prescriptions (2-15 per admission)

## Features

- **Realistic distributions**: Preserves statistical patterns from original data
- **Medical consistency**: Maintains realistic relationships between clinical variables
- **OUD cases**: Includes ~5% of admissions with Opioid Use Disorder diagnoses
- **Drug variety**: Includes opioids, benzodiazepines, antibiotics, and other medication classes
- **Temporal realism**: Realistic admission dates, lengths of stay, and prescription timing

## Data Quality

The synthetic data maintains:
- Proper foreign key relationships (subject_id, hadm_id)
- Realistic medical terminology and codes
- Appropriate statistical distributions
- Clinical plausibility (e.g., prescription dates within admission dates)

## Note

This is synthetic data for research and development purposes only. It should not be used for clinical decision making or real patient care.