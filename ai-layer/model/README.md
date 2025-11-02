# Two-Model Opioid Audit System - Model Training

This directory contains scripts for training and comparing models for the opioid prescription audit system.

## System Architecture

### Two-Model Approach

The audit system uses **two independent models** to flag potentially inappropriate opioid prescriptions:

1. **Eligibility Model**: Predicts clinical need for opioids
   - Target: `opioid_eligibility` (based on pain diagnosis codes)
   - Features: 8 features **EXCLUDING** opioid prescription history
   - Purpose: Determine if prescription is clinically justified
   - No data leakage from past prescriptions

2. **OUD Risk Model**: Predicts Opioid Use Disorder risk
   - Target: `y_oud` (from ICD diagnosis codes)
   - Features: 11 features **INCLUDING** opioid exposure patterns
   - Purpose: Identify patients at high risk for OUD
   - Opioid history is a legitimate predictor

### Audit Logic

```
Flag prescription if:
  (Eligibility Model predicts NO clinical need) OR
  (OUD Risk Model predicts HIGH risk)
```

## Model Comparison

For each of the two models, we train and compare:

- **PyCaret Models**: Automatically tests multiple traditional ML algorithms
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (GBC)
  - AdaBoost
  - Extra Trees
  - LightGBM
  - XGBoost
  - Selects best performer based on AUC
  - Tunes hyperparameters automatically

- **Deep Neural Networks**: Custom PyTorch models
  - Architecture: [Input → 128 → 64 → 32 → 16 → Output]
  - BatchNormalization + ReLU + Dropout
  - Early stopping on validation AUC
  - Class weights for imbalanced data

## Training Workflow

### Step 1: Train PyCaret Models

```bash
cd ai-layer/model

# Train both models using PyCaret
python train_two_models_pycaret.py --data-dir ../processed_data/1000 --output-dir ./results
```

This will:
- Load SHAP-selected features from `processed_data/1000/`
- Train Eligibility Model (8 features, no opioid history)
- Train OUD Risk Model (11 features, includes opioid patterns)
- Compare 7 different ML algorithms for each model
- Tune hyperparameters using validation set
- Evaluate on test set
- Save models, predictions, and metrics to `./results/`

**Expected Output:**
```
results/
  pycaret_eligibility_model.pkl
  pycaret_eligibility_model_predictions.csv
  pycaret_eligibility_model_metrics.json
  pycaret_oud_risk_model.pkl
  pycaret_oud_risk_model_predictions.csv
  pycaret_oud_risk_model_metrics.json
```

### Step 2: Train Deep Neural Networks

```bash
# Train both models using DNNs
python train_two_models_dnn.py --data-dir ../processed_data/1000 --output-dir ./results
```

This will:
- Load same SHAP-selected features
- Build and train DNNs for both models
- Use early stopping (patience=15 epochs)
- Apply class weights for imbalanced OUD data
- Evaluate on test set
- Save models, predictions, and metrics

**Expected Output:**
```
results/
  dnn_eligibility_model.pth
  dnn_eligibility_model_predictions.csv
  dnn_eligibility_model_metrics.json
  dnn_oud_risk_model.pth
  dnn_oud_risk_model_predictions.csv
  dnn_oud_risk_model_metrics.json
```

### Step 3: Compare All Models

```bash
# Compare PyCaret vs DNN for both models
python compare_all_models.py --results-dir ./results
```

This will:
- Load all metrics and predictions
- Compare performance on 5 metrics (Accuracy, Precision, Recall, F1, AUC)
- Show confusion matrices
- Determine winner for each model
- Generate comprehensive comparison report

**Expected Output:**
```
results/
  model_comparison_report.txt  <- Summary of best models
```

## Data Requirements

### Input Files (from feature selection)

```
processed_data/1000/
  train_data.csv           # 567 samples
  validation_data.csv      # 63 samples
  test_data.csv            # 300 samples
  metadata.json            # Feature information
```

### Metadata Structure

```json
{
  "eligibility_features": [
    "age_at_first_admit",
    "insurance",
    "avg_los_days",
    "atc_A_rx_count",
    "total_los_days",
    "n_hospital_admits",
    "atc_H_rx_count",
    "atc_C_rx_count"
  ],
  "oud_features": [
    "atc_A_rx_count",
    "atc_H_rx_count",
    "atc_Other_rx_count",
    "atc_J_rx_count",
    "atc_N_rx_count",
    "atc_C_rx_count",
    "age_at_first_admit",
    "n_hospital_admits",
    "total_los_days",
    "atc_R_rx_count",
    "opioid_rx_count"
  ],
  "oud_positive_train": 16,
  "oud_positive_validation": 2,
  "oud_positive_test": 8,
  "eligibility_positive_train": 405,
  "eligibility_positive_validation": 43,
  "eligibility_positive_test": 192
}
```

## Key Features

### Eligibility Model Features (8)

**No opioid prescription history** - prevents data leakage:

1. `age_at_first_admit`: Patient age
2. `insurance`: Insurance type
3. `avg_los_days`: Average length of hospital stay
4. `total_los_days`: Total days hospitalized
5. `n_hospital_admits`: Number of admissions
6. `atc_A_rx_count`: Alimentary/metabolism drugs
7. `atc_H_rx_count`: Hormonal drugs
8. `atc_C_rx_count`: Cardiovascular drugs

### OUD Risk Model Features (11)

**Includes opioid exposure** - legitimate predictor for OUD:

1. `opioid_rx_count`: **Number of opioid prescriptions**
2. `age_at_first_admit`: Patient age
3. `n_hospital_admits`: Number of admissions
4. `total_los_days`: Total days hospitalized
5. `atc_N_rx_count`: Nervous system drugs
6. `atc_J_rx_count`: Anti-infectives
7. `atc_A_rx_count`: Alimentary drugs
8. `atc_C_rx_count`: Cardiovascular drugs
9. `atc_R_rx_count`: Respiratory drugs
10. `atc_H_rx_count`: Hormonal drugs
11. `atc_Other_rx_count`: Other medications

## Performance Metrics

All models are evaluated on:

- **Accuracy**: Overall correct predictions
- **Precision**: Of flagged cases, how many are truly inappropriate
- **Recall**: Of truly inappropriate cases, how many are flagged
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (primary metric)

### Interpretation

For the audit system:
- **High Precision**: Few false alarms (good for provider trust)
- **High Recall**: Few missed inappropriate prescriptions (good for patient safety)
- **High AUC**: Good discrimination between appropriate and inappropriate

## Example Output

### PyCaret Training (Eligibility Model)

```
================================================================================
TRAINING MODEL: eligibility_model
Target: opioid_eligibility
================================================================================

Class distribution:
  Training:   {0: 162, 1: 405} ({0: 0.286, 1: 0.714})
  Validation: {0: 20, 1: 43} ({0: 0.317, 1: 0.683})
  Test:       {0: 108, 1: 192} ({0: 0.36, 1: 0.64})

Features (8): ['age_at_first_admit', 'insurance', 'avg_los_days', ...]

Comparing models...
  Models: Logistic Regression, Random Forest, Gradient Boosting...

✓ Best model: LightGBM Classifier

Tuning hyperparameters...
  Optimization metric: AUC
  Iterations: 30

================================================================================
FINAL EVALUATION ON TEST SET (300 samples)
================================================================================

Test Set Performance:
  Accuracy:  0.7533
  Precision: 0.8261
  Recall:    0.7917
  F1 Score:  0.8085
  ROC AUC:   0.8245

Confusion Matrix:
  TN:     85  |  FP:     23
  FN:     51  |  TP:    141

✓ Model saved: ./results/pycaret_eligibility_model.pkl
```

### DNN Training (OUD Risk Model)

```
================================================================================
TRAINING DNN: oud_risk_model
Target: y_oud
================================================================================

Class distribution:
  Training:   {0: 551, 1: 16} ({0: 0.972, 1: 0.028})
  Validation: {0: 61, 1: 2} ({0: 0.968, 1: 0.032})
  Test:       {0: 292, 1: 8} ({0: 0.973, 1: 0.027})

Features (11): ['atc_A_rx_count', 'atc_H_rx_count', 'opioid_rx_count', ...]

Class weights: [0.512, 17.234]

Model architecture:
  Input: 11 features
  Hidden layers: [128, 64, 32, 16]
  Output: 1 (binary classification)
  Total parameters: 26,241

Training Deep Neural Network...
  Epochs: 100
  Early stopping patience: 15
  Device: cpu

  Epoch   1/100: Train Loss: 0.3421 | Val Loss: 0.2156 | Val AUC: 0.7812 | Val Acc: 0.9683
  Epoch  10/100: Train Loss: 0.1234 | Val Loss: 0.1823 | Val AUC: 0.8594 | Val Acc: 0.9683
  ...
  
  Early stopping at epoch 43
  Best validation AUC: 0.8906

================================================================================
FINAL EVALUATION ON TEST SET
================================================================================

Test Set Performance:
  Accuracy:  0.9733
  Precision: 0.7500
  Recall:    0.7500
  F1 Score:  0.7500
  ROC AUC:   0.8828

Confusion Matrix:
  TN:    290  |  FP:      2
  FN:      6  |  TP:      2

✓ Model saved: ./results/dnn_oud_risk_model.pth
```

### Model Comparison

```
================================================================================
TWO-MODEL OPIOID AUDIT SYSTEM - MODEL COMPARISON
================================================================================

ELIGIBILITY MODEL
================================================================================

╔═══════════╦════════════╦════════════╦══════════════╗
║ Metric    ║ PyCaret    ║ DNN        ║ Winner       ║
╠═══════════╬════════════╬════════════╬══════════════╣
║ ACCURACY  ║ 0.7533     ║ 0.7600     ║ DNN →        ║
║ PRECISION ║ 0.8261     ║ 0.8182     ║ PyCaret ←    ║
║ RECALL    ║ 0.7917     ║ 0.8125     ║ DNN →        ║
║ F1        ║ 0.8085     ║ 0.8153     ║ DNN →        ║
║ AUC       ║ 0.8245     ║ 0.8312     ║ DNN →        ║
╚═══════════╩════════════╩════════════╩══════════════╝

Score Summary:
  PyCaret wins: 1
  DNN wins:     4
  Ties:         0

✓ Overall winner: DNN

OUD RISK MODEL
================================================================================

╔═══════════╦════════════╦════════════╦══════════════╗
║ Metric    ║ PyCaret    ║ DNN        ║ Winner       ║
╠═══════════╬════════════╬════════════╬══════════════╣
║ ACCURACY  ║ 0.9700     ║ 0.9733     ║ DNN →        ║
║ PRECISION ║ 0.7143     ║ 0.7500     ║ DNN →        ║
║ RECALL    ║ 0.6250     ║ 0.7500     ║ DNN →        ║
║ F1        ║ 0.6667     ║ 0.7500     ║ DNN →        ║
║ AUC       ║ 0.8750     ║ 0.8828     ║ DNN →        ║
╚═══════════╩════════════╩════════════╩══════════════╝

Score Summary:
  PyCaret wins: 0
  DNN wins:     5
  Ties:         0

✓ Overall winner: DNN
```

## Scaling to Production

Once you've validated the approach on 1000 patients:

### 1. Generate Larger Dataset

```bash
# Generate 100K patients (takes ~30-60 minutes)
cd data-processing/data_generation
python generate_synthetic_data.py --patients 100000 --output-dir ../../synthetic_data/100000/mimic-iv-synthetic
```

### 2. Run Feature Selection on 100K

```bash
cd ai-layer/feature_selection
python shap_feature_selection.py --input-dir ../../synthetic_data/100000/mimic-iv-synthetic --output-dir ../processed_data/100000 --temp-dir temp_data_100000
```

### 3. Train Models on 100K

```bash
cd ai-layer/model

# PyCaret
python train_two_models_pycaret.py --data-dir ../processed_data/100000 --output-dir ./results_100k

# DNN
python train_two_models_dnn.py --data-dir ../processed_data/100000 --output-dir ./results_100k

# Compare
python compare_all_models.py --results-dir ./results_100k
```

## Troubleshooting

### Issue: OUD model has poor recall

**Cause**: Severe class imbalance (2.8% OUD rate)

**Solutions**:
- DNN uses class weights automatically
- PyCaret: Set `fix_imbalance=True` in setup()
- Adjust decision threshold (default 0.5 → 0.3 for higher recall)

### Issue: Training takes too long

**Solutions**:
- Reduce PyCaret models: Remove slow algorithms
- Reduce DNN epochs: Set max to 50 instead of 100
- Use GPU: If available, DNN will use it automatically

### Issue: Models overfit

**Symptoms**: High train performance, low test performance

**Solutions**:
- DNN: Increase dropout rate (0.3 → 0.5)
- PyCaret: Reduce n_iter for tuning (30 → 10)
- Get more data: Scale to 100K patients

## Next Steps

After training and comparing models:

1. **Select Best Models**: Choose best performer for each task
2. **Implement Audit System**: Combine both models with OR logic
3. **Deploy**: Create prediction API or batch processor
4. **Monitor**: Track performance on real prescription data
5. **Retrain**: Periodically update models with new data

## File Structure

```
model/
  train_two_models_pycaret.py    # PyCaret training
  train_two_models_dnn.py         # DNN training
  compare_all_models.py           # Model comparison
  README.md                       # This file
  
  results/                        # Training outputs (1K patients)
    pycaret_*.pkl                 # Saved PyCaret models
    pycaret_*_predictions.csv     # Test predictions
    pycaret_*_metrics.json        # Performance metrics
    dnn_*.pth                     # Saved PyTorch models
    dnn_*_predictions.csv         # Test predictions
    dnn_*_metrics.json            # Performance metrics
    model_comparison_report.txt   # Comparison summary
  
  results_100k/                   # Training outputs (100K patients)
    [same structure as results/]

  oldfiles/                       # Legacy single-target models
    pycaret_model_oud_with_shap.py
    deep_neural_network_oud.py
    compare_models.py
```

## References

- PyCaret Documentation: https://pycaret.org/
- PyTorch Documentation: https://pytorch.org/
- SHAP Feature Selection: See `../feature_selection/README.md`
