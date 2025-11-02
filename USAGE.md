# Command-Line Usage Guide

## Synthetic Data Generation

### Quick Start
```powershell
# Fast testing (1K patients, ~10 seconds)
python generate_synthetic_data.py --patients 1000

# Medium dataset (10K patients, ~1 minute)
python generate_synthetic_data.py --patients 10000

# Full dataset (100K patients, ~10 minutes)
python generate_synthetic_data.py --patients 100000
```

### Custom Directories
```powershell
# Generate 100K to separate directory (for parallel work)
python generate_synthetic_data.py --patients 100000 --output-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp

# Use custom source data
python generate_synthetic_data.py --patients 5000 --source-dir /path/to/mimic/data
```

### All Options
```
--patients, -n      Number of patients to generate (default: 1000)
--source-dir, -s    Directory with original MIMIC-IV demo data
--output-dir, -o    Directory to save generated synthetic data
```

## Feature Selection

### Quick Start
```powershell
cd ../../ai-layer/feature_selection

# Process default 1K dataset
python shap_feature_selection.py

# Process 100K dataset
python shap_feature_selection.py --input-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp --output-dir ../processed_data_100k --temp-dir temp_data_100k
```

### All Options
```
--input-dir, -i     Directory containing synthetic data files
--output-dir, -o    Directory to save processed datasets
--temp-dir, -t      Directory for intermediate files
```

## Workflow Examples

### Development Workflow (Fast)
```powershell
# 1. Generate small dataset
cd data-processing/data_generation
python generate_synthetic_data.py --patients 1000

# 2. Validate data quality
cd ../..
python check_synthetic_data.py

# 3. Run feature selection
cd ai-layer/feature_selection
python shap_feature_selection.py

# 4. Train models (next step)
```

### Production Workflow (Background)
```powershell
# 1. Generate large dataset in background
cd data-processing/data_generation
Start-Process powershell -ArgumentList "-Command", "& 'C:\Users\unell\OneDrive\Desktop\AutoRxAudit\.venv\Scripts\python.exe' generate_synthetic_data.py --patients 100000 --output-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp"

# 2. Continue working with 1K dataset
python generate_synthetic_data.py --patients 1000

# 3. When 100K completes, process it
cd ../../ai-layer/feature_selection
python shap_feature_selection.py --input-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp --output-dir ../processed_data_100k --temp-dir temp_data_100k
```

## Data Validation
```powershell
# Check synthetic data quality
cd C:\Users\unell\OneDrive\Desktop\AutoRxAudit\AutoRxAudit
python check_synthetic_data.py
```

## Tips

1. **Start small**: Always test with `--patients 1000` first
2. **Parallel work**: Generate 100K in background while working with 1K
3. **Use different directories**: Keep datasets separate with `--output-dir`
4. **Check quality**: Run `check_synthetic_data.py` before feature selection
5. **Monitor progress**: Scripts show progress bars and completion stats
