# Test API Scripts

This folder contains test and validation scripts for the AutoRxAudit API.

## Scripts

### validate_api.py
Main validation script that tests the API with both opioid and non-opioid prescriptions.
- Tests 5 low-risk patients with non-opioid (Acetaminophen) - should NOT be flagged
- Tests 5 low-risk patients with opioid (Oxycodone) - risk assessment based on patient history
- Generates detailed output with eligibility and OUD risk scores

**Usage:**
```bash
cd singleapp/api
python test_api/validate_api.py
```

### find_low_risk_patients.py
Queries the database to identify patients with minimal OUD risk indicators.
- Searches for patients with: no opioid history, minimal admissions, low severity
- Used to identify good test cases for validation

**Usage:**
```bash
cd singleapp/api
python test_api/find_low_risk_patients.py
```

### check_patients.py
Simple utility to list available patient IDs from the database.
- Returns first 20 patient IDs
- Useful for finding valid patient IDs for testing

**Usage:**
```bash
cd singleapp/api
python test_api/check_patients.py
```

### check_db_schema.py
Diagnostic script to inspect the database schema.
- Shows prescriptions table columns
- Displays sample prescription record
- Checks for atc_code column availability

**Usage:**
```bash
cd singleapp/api
python test_api/check_db_schema.py
```

### test_model_directly.py
Direct model testing bypassing the API.
- Loads models from checkpoints
- Calculates features for test patients
- Shows raw model predictions without API layer

**Usage:**
```bash
cd singleapp/api
python test_api/test_model_directly.py
```

### FEATURE_COMPARISON.py
Documentation comparing feature calculations between training and API.
- Maps all 16 eligibility features and 19 OUD features
- Shows training logic vs API implementation
- Identifies mismatches that need fixing

## Running Tests

The API server must be running to use validate_api.py:

1. Start the API server (press F5 in VS Code or run manually)
2. Run validation: `python test_api/validate_api.py`

All other scripts are standalone and don't require the API to be running.

## API Endpoints Tested

- `POST /audit-prescription` - Main prescription auditing endpoint
- Response includes: flagged status, scores, predictions, explanations

## Expected Results

**Non-opioid prescriptions:** Should NOT be flagged
- Eligibility: 1.000
- OUD: 0.000
- Flagged: False

**Opioid prescriptions:** Risk assessment varies by patient
- Eligibility: 0.0-1.0 (based on pain diagnosis)
- OUD: 0.0-1.0 (based on patient risk factors)
- Flagged: True if NOT eligible OR high OUD risk
