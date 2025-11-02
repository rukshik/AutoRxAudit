# Miscellaneous Files

This folder contains deprecated scripts and utilities that are no longer part of the active codebase but are kept for reference.

## Deprecated Model Scripts

### Old Single-Model Approaches
- **deep_neural_network_oud.py** - Early DNN implementation (superseded by `train_two_models_dnn.py`)
- **pycaret_model_oud_with_shap.py** - Early PyCaret implementation with SHAP (superseded)
- **pycaret_model_wrapper.py** - Wrapper for PyCaret models (no longer needed)

## Deprecated Batch Processing Scripts

### Old Data Processing Automation
- **process_all_datasets.ps1** - PowerShell batch script for processing 10K/50K/100K datasets (v2)
- **process_all_datasets.sh** - Bash equivalent batch script (v2)

These scripts automated SHAP feature selection across multiple dataset sizes but reference:
- Old data locations (`synthetic_data/` instead of `datasets/`)
- Old v2 feature approach (missing BMI, DRG, ICU correlation)
- Moved script locations (`check_oud_signals.py` now in `data-processing/scripts/`)

For current workflows, see `SESSION_JOURNAL.md` for command documentation.

## Why These Were Deprecated

### Evolution to Two-Model System
The project evolved from a single OUD risk model to a **two-model audit system**:

1. **Eligibility Model** - Predicts clinical need for opioids (excludes opioid history)
2. **OUD Risk Model** - Predicts Opioid Use Disorder risk (includes opioid exposure)

**Audit Logic:** Flag prescription if (No Clinical Need) OR (High OUD Risk)

### Current Production Scripts
The active training scripts are:
- `ai-layer/model/dnn_models.py` - DNN two-model system
- `ai-layer/model/pycaret_models.py` - PyCaret two-model system

## Historical Context

These files represent earlier iterations of the project:
- **Single-task learning** (OUD risk only) → **Multi-task learning** (Eligibility + OUD)
- **Simple feature engineering** → **Pain-correlated features** (BMI, DRG severity, ICU)
- **Basic SHAP selection** → **Two-stage SHAP** (separate for each model)

## Note

These files are kept for:
1. **Reference** - Understanding project evolution
2. **Code reuse** - Potential code snippets for future work
3. **Comparison** - Benchmarking against older approaches

Do not use these files for new development. Refer to the main training scripts in `ai-layer/model/`.
