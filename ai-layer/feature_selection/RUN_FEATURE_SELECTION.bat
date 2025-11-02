@echo off
echo ================================================================================
echo Running SHAP Feature Selection for Two-Model Opioid Audit System
echo ================================================================================
echo.
echo This will perform feature selection for:
echo   - Model 1: Eligibility (clinical need for opioids)
echo   - Model 2: OUD Risk (preventive risk assessment)
echo.
echo Estimated time: 5-10 minutes for 100K patients
echo ================================================================================
echo.

REM Activate virtual environment and run script
cd /d "%~dp0"
call ..\..\..\..\.venv\Scripts\activate.bat
python shap_feature_selection.py

echo.
echo ================================================================================
echo Feature selection complete! Check ai-layer/processed_data/ for outputs.
echo ================================================================================
pause
