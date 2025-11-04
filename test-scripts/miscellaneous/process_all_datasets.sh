#!/bin/bash
# Process all datasets: feature selection and signal checking

echo "================================"
echo "Processing 10K dataset..."
echo "================================"
cd ai-layer/feature_selection
python shap_feature_selection.py --input-dir ../../synthetic_data/10000/mimic-iv-synthetic --output-dir ../processed_data/10000 --temp-dir temp_data_10000
python check_oud_signals.py 10000

echo ""
echo "================================"
echo "Processing 50K dataset..."
echo "================================"
python shap_feature_selection.py --input-dir ../../synthetic_data/50000/mimic-iv-synthetic --output-dir ../processed_data/50000 --temp-dir temp_data_50000
python check_oud_signals.py 50000

echo ""
echo "================================"
echo "Processing 100K dataset..."
echo "================================"
python shap_feature_selection.py --input-dir ../../synthetic_data/100000/mimic-iv-synthetic --output-dir ../processed_data/100000 --temp-dir temp_data_100000
python check_oud_signals.py 100000

echo ""
echo "================================"
echo "All datasets processed!"
echo "================================"
