# Feature Selection Module

This module performs SHAP-based feature selection on synthetic MIMIC-IV data for OUD prediction.

## Purpose

- Read synthetic patient data from `../synthetic_data/`
- Engineer clinical features (demographics, prescriptions, diagnoses)
- Perform SHAP analysis to identify most important features
- Save processed datasets to `../processed_data/` for main model training

## Structure

```
ai-layer/
├── feature_selection/
│   ├── shap_feature_selection.py    # Main feature selection script
│   ├── temp_data/                    # SHAP analysis artifacts (temporary)
│   │   ├── shap_subset.csv          # Data subset used for SHAP
│   │   ├── shap_feature_importance.csv
│   │   ├── shap_importance_oud.csv
│   │   └── shap_importance_rx.csv
│   └── README.md
└── processed_data/                   # Final processed datasets for model
    ├── train_data.csv
    ├── validation_data.csv
    ├── test_data.csv
    ├── full_data_selected_features.csv
    └── metadata.json
```

## Usage

```powershell
# From AutoRxAudit directory
cd ai-layer\feature_selection
..\..\..\.venv\Scripts\python.exe shap_feature_selection.py
```

## Data Flow

1. **Input**: `../../synthetic_data/mimic-clinical-iv-demo/hosp/` (50K synthetic patients)
2. **Processing**: 
   - Feature engineering
   - Data splitting (7% SHAP, 56% train, 7% validation, 30% test)
   - SHAP analysis on subset
   - Feature selection
3. **Temporary Output**: `temp_data/` (SHAP analysis artifacts)
   - `shap_subset.csv` - Data used for SHAP analysis
   - `shap_feature_importance.csv` - Selected features summary
   - `shap_importance_oud.csv` - Detailed importance for OUD prediction
   - `shap_importance_rx.csv` - Detailed importance for opioid prescription
4. **Final Output**: `../processed_data/`
   - `train_data.csv` - Training data with selected features (28,350 records)
   - `validation_data.csv` - Validation data for hyperparameter tuning (3,150 records)
   - `test_data.csv` - Test data for final evaluation (15,000 records)
   - `full_data_selected_features.csv` - Complete dataset
   - `metadata.json` - Dataset metadata

## Features Analyzed

- **Demographics**: age, gender, race, insurance
- **Hospital Utilization**: admission counts, length of stay
- **Prescriptions**: opioid exposure, benzodiazepine use, ATC drug classes
- **Target Labels**: OUD diagnosis (ICD-9/10), opioid prescription flag

## SHAP Analysis

Uses TreeExplainer with Random Forest to calculate:
- Feature importance for OUD prediction
- Feature importance for opioid prescription prediction
- Combines top 15 features from each analysis

## Output for Main Model

The `ai-layer/processed_data/` directory contains:
- **train_data.csv**: For training the neural network (56.7% of data)
- **validation_data.csv**: For hyperparameter tuning, early stopping, model selection (6.3% of data)
- **test_data.csv**: For final model evaluation only (30% of data)
- Only selected features (reduced dimensionality: 21 → 17 features)
- Proper data separation to prevent leakage
- Metadata for reproducibility

### Data Usage Guidelines
- **Training Set**: Use for model learning (gradient descent, backpropagation)
- **Validation Set**: Use during training for:
  - Hyperparameter tuning
  - Early stopping decisions
  - Model architecture selection
  - Monitor overfitting
- **Test Set**: Use ONLY ONCE after all development is complete for final performance evaluation
