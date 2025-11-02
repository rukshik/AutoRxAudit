@echo off
REM ============================================================================
REM Quick Commands for AutoRxAudit Data Generation
REM ============================================================================

echo Available commands:
echo.
echo 1. Generate 1K patients (fast - for testing)
echo    python generate_synthetic_data.py --patients 1000
echo.
echo 2. Generate 10K patients (medium)
echo    python generate_synthetic_data.py --patients 10000
echo.
echo 3. Generate 100K patients (full dataset - slow)
echo    python generate_synthetic_data.py --patients 100000
echo.
echo 4. Generate 100K patients to separate directory (for background)
echo    python generate_synthetic_data.py --patients 100000 --output-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp
echo.
echo 5. Run feature selection on 1K dataset
echo    cd ..\..\..\ai-layer\feature_selection
echo    python shap_feature_selection.py
echo.
echo 6. Run feature selection on 100K dataset
echo    cd ..\..\..\ai-layer\feature_selection
echo    python shap_feature_selection.py --input-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp --output-dir ../processed_data_100k --temp-dir temp_data_100k
echo.
echo ============================================================================
pause
