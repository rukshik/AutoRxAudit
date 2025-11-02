# Process all datasets: feature selection and signal checking

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Processing 10K dataset..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Set-Location ai-layer\feature_selection
python shap_feature_selection.py --input-dir ../../synthetic_data/10000/mimic-iv-synthetic --output-dir ../processed_data/10000 --temp-dir temp_data_10000
python check_oud_signals.py 10000

Write-Host ""
Write-Host "================================" -ForegroundColor Yellow
Write-Host "Processing 50K dataset..." -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow
python shap_feature_selection.py --input-dir ../../synthetic_data/50000/mimic-iv-synthetic --output-dir ../processed_data/50000 --temp-dir temp_data_50000
python check_oud_signals.py 50000

Write-Host ""
Write-Host "================================" -ForegroundColor Magenta
Write-Host "Processing 100K dataset..." -ForegroundColor Magenta
Write-Host "================================" -ForegroundColor Magenta
python shap_feature_selection.py --input-dir ../../synthetic_data/100000/mimic-iv-synthetic --output-dir ../processed_data/100000 --temp-dir temp_data_100000
python check_oud_signals.py 100000

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "All datasets processed!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

Set-Location ..\..\
