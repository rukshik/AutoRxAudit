"""
FEATURE CALCULATION COMPARISON
Training (shap_feature_selection.py) vs API (feature_calculator.py)

This document maps how each feature is calculated in training vs API to ensure consistency.
"""

# =============================================================================
# ELIGIBILITY MODEL FEATURES (16 features)
# =============================================================================

"""
From checkpoint: eligibility model uses these features:
'avg_drg_severity', 'bmi', 'avg_drg_mortality', 'n_icu_admissions',
'n_icu_stays', 'max_drg_severity', 'high_severity_flag', 'total_icu_hours',
'obesity_flag', 'total_icu_days', 'atc_A_rx_count', 'n_admissions_with_drg',
'n_hospital_admits', 'avg_los_days', 'total_los_days', 'has_bmi'
"""

# 1. avg_drg_severity
# TRAINING: drgcodes.groupby("subject_id").agg(avg_drg_severity=("drg_severity", "mean"))
# API: SELECT AVG(drg_severity::float) FROM drgcodes WHERE patient_id = %s
# STATUS: ✓ MATCH

# 2. bmi
# TRAINING: omr[omr["result_name"] == "BMI (kg/m2)"].groupby("subject_id").agg(bmi=("bmi_value", "mean"))
# API: SELECT AVG(result_value::float) FROM omr WHERE patient_id = %s AND result_name = 'BMI (kg/m2)'
# STATUS: ✓ MATCH

# 3. avg_drg_mortality
# TRAINING: drgcodes.groupby("subject_id").agg(avg_drg_mortality=("drg_mortality", "mean"))
# API: SELECT AVG(drg_mortality::float) FROM drgcodes WHERE patient_id = %s
# STATUS: ✓ MATCH

# 4. n_icu_admissions
# TRAINING: icu_transfers.groupby("subject_id").agg(n_icu_admissions=("hadm_id", "nunique"))
#           where icu_transfers = transfers[transfers["careunit"].str.contains("ICU", case=False)]
# API: SELECT COUNT(DISTINCT admission_id) FROM transfers WHERE patient_id = %s AND care_unit ILIKE '%ICU%'
# STATUS: ✓ MATCH

# 5. n_icu_stays
# TRAINING: icu_transfers.groupby("subject_id").agg(n_icu_stays=("transfer_id", "count"))
# API: SELECT COUNT(*) FROM transfers WHERE patient_id = %s AND care_unit ILIKE '%ICU%'
# STATUS: ✓ MATCH

# 6. max_drg_severity
# TRAINING: drgcodes.groupby("subject_id").agg(max_drg_severity=("drg_severity", "max"))
# API: SELECT MAX(drg_severity::float) FROM drgcodes WHERE patient_id = %s
# STATUS: ✓ MATCH (if implemented)

# 7. high_severity_flag
# TRAINING: (drg_agg["max_drg_severity"] >= 3).astype(int)
# API: Derived from max_drg_severity >= 3
# STATUS: ✓ MATCH

# 8. total_icu_hours
# TRAINING: Sum of (outtime - intime).total_seconds() / 3600 for ICU transfers
# API: SELECT SUM(EXTRACT(EPOCH FROM (out_time - in_time)) / 3600) FROM transfers WHERE careunit ILIKE '%ICU%'
# STATUS: ✓ MATCH

# 9. obesity_flag
# TRAINING: (bmi_agg["bmi"] > 30).astype(int)
# API: Derived from bmi > 30
# STATUS: ✓ MATCH

# 10. total_icu_days
# TRAINING: total_icu_hours / 24.0
# API: total_icu_hours / 24
# STATUS: ✓ MATCH

# 11. atc_A_rx_count
# TRAINING: Count of non-opioid, non-benzo prescriptions where drug name matches ATC-A patterns
#           Patterns: ['insulin', 'metformin', 'glipizide', 'lantus', 'proton pump inhibitor', 'prazole', 'omeprazole']
# API: NEEDS UPDATE - currently uses atc_code LIKE 'A%', should use pattern matching
# STATUS: ❌ MISMATCH - NEEDS FIX

# 12. n_admissions_with_drg
# TRAINING: drgcodes.groupby("subject_id").agg(n_admissions_with_drg=("hadm_id", "nunique"))
# API: SELECT COUNT(DISTINCT admission_id) FROM drgcodes WHERE patient_id = %s
# STATUS: ✓ MATCH

# 13. n_hospital_admits
# TRAINING: admissions.groupby("subject_id").agg(n_hospital_admits=("hadm_id", "nunique"))
# API: SELECT COUNT(*) FROM admissions WHERE patient_id = %s
# STATUS: ✓ MATCH

# 14. avg_los_days
# TRAINING: admissions.groupby("subject_id").agg(avg_los_days=("los", "mean"))
#           where los = (dischtime - admittime).dt.total_seconds() / 86400.0
# API: SELECT AVG(EXTRACT(EPOCH FROM (discharge_time - admit_time)) / 86400) FROM admissions
# STATUS: ✓ MATCH

# 15. total_los_days
# TRAINING: admissions.groupby("subject_id").agg(total_los_days=("los", "sum"))
# API: SELECT SUM(EXTRACT(EPOCH FROM (discharge_time - admit_time)) / 86400) FROM admissions
# STATUS: ✓ MATCH

# 16. has_bmi
# TRAINING: (bmi_agg["has_bmi"] > 0).astype(int) where has_bmi = count of BMI records
# API: Derived from BMI record existence
# STATUS: ✓ MATCH


# =============================================================================
# OUD MODEL FEATURES (19 features)
# =============================================================================

"""
From checkpoint: OUD model uses these features:
'opioid_rx_count', 'distinct_opioids', 'opioid_hadms', 'opioid_exposure_days',
'n_icu_stays', 'total_icu_hours', 'age_at_first_admit', 'n_icu_admissions',
'total_icu_days', 'avg_drg_mortality', 'any_opioid_flag', 'atc_J_rx_count',
'bmi', 'avg_drg_severity', 'avg_los_days', 'max_drg_mortality',
'atc_Other_rx_count', 'atc_C_rx_count', 'atc_N_rx_count'
"""

# 1. opioid_rx_count
# TRAINING: Count of prescriptions where drug matches OPIOID_PATTERNS
#           ['morphine', 'hydromorphone', 'oxycodone', 'hydrocodone', 'fentanyl', 'codeine', 
#            'tramadol', 'oxymorphone', 'tapentadol', 'methadone', 'buprenorphine']
# API: SELECT COUNT(*) FROM prescriptions WHERE patient_id = %s AND LOWER(drug_name) LIKE any opioid pattern
# STATUS: ✓ MATCH

# 2. distinct_opioids
# TRAINING: opioid_df.groupby("subject_id").agg(distinct_opioids=("drug", "nunique"))
# API: SELECT COUNT(DISTINCT drug_name) FROM prescriptions WHERE ... opioid patterns
# STATUS: ✓ MATCH

# 3. opioid_hadms
# TRAINING: opioid_df.groupby("subject_id").agg(opioid_hadms=("hadm_id", "nunique"))
# API: SELECT COUNT(DISTINCT admission_id) FROM prescriptions WHERE ... opioid patterns
# STATUS: ✓ MATCH

# 4. opioid_exposure_days
# TRAINING: Sum of (stoptime - starttime).dt.total_seconds() / 86400.0 for opioid prescriptions
# API: SELECT SUM(EXTRACT(EPOCH FROM (stop_time - start_time)) / 86400) FROM prescriptions WHERE ... opioids
# STATUS: ✓ MATCH

# 5-10. ICU features, DRG features, demographics
# STATUS: Same as eligibility model - ✓ MATCH

# 11. any_opioid_flag
# TRAINING: (opioid_rx_count > 0).astype(int)
# API: Derived from opioid_rx_count > 0
# STATUS: ✓ MATCH

# 12. atc_J_rx_count
# TRAINING: Count of non-opioid, non-benzo prescriptions where drug name matches ATC-J patterns
#           Patterns: ['antibiotic', 'penicillin', 'cephalosporin', 'vancomycin', 'ciprofloxacin', 
#                      'azithromycin', 'amoxicillin']
# API: NEEDS UPDATE - currently uses atc_code LIKE 'J%', should use pattern matching
# STATUS: ❌ MISMATCH - NEEDS FIX

# 13. bmi
# STATUS: Same as eligibility - ✓ MATCH

# 14. avg_drg_severity, 15. avg_los_days
# STATUS: Same as eligibility - ✓ MATCH

# 16. max_drg_mortality
# TRAINING: drgcodes.groupby("subject_id").agg(max_drg_mortality=("drg_mortality", "max"))
# API: SELECT MAX(drg_mortality::float) FROM drgcodes WHERE patient_id = %s
# STATUS: ✓ MATCH (just implemented)

# 17. atc_Other_rx_count
# TRAINING: Count of non-opioid, non-benzo prescriptions that DON'T match any ATC category (A,B,C,H,J,N,R)
# API: NEEDS UPDATE - should use pattern matching logic
# STATUS: ❌ MISMATCH - NEEDS FIX (just implemented but needs testing)

# 18. atc_C_rx_count
# TRAINING: Count of non-opioid, non-benzo prescriptions where drug name matches ATC-C patterns
#           Patterns: ['antihypertensive', 'beta blocker', 'metoprolol', 'carvedilol', 'ace inhibitor',
#                      'lisinopril', 'statin', 'atorvastatin', 'simvastatin', 'amlodipine', 'furosemide']
# API: NEEDS UPDATE - currently uses atc_code LIKE 'C%', should use pattern matching
# STATUS: ❌ MISMATCH - NEEDS FIX

# 19. atc_N_rx_count
# TRAINING: Count of non-opioid, non-benzo prescriptions where drug name matches ATC-N patterns
#           Patterns: ['antidepressant', 'ssri', 'snri', 'sertraline', 'antipsychotic', 'risperidone',
#                      'quetiapine', 'haloperidol', 'anticonvulsant']
# API: NEEDS UPDATE - currently uses atc_code LIKE 'N%', should use pattern matching
# STATUS: ❌ MISMATCH - NEEDS FIX


# =============================================================================
# CRITICAL FIXES NEEDED
# =============================================================================

"""
The API's atc_rx_count methods are already fixed to use pattern matching.
Need to verify they work correctly in the live API.

All ATC category calculations (A, B, C, H, J, N, R, Other) should:
1. Exclude opioid prescriptions (11 patterns)
2. Exclude benzo prescriptions (8 patterns)
3. Match drug_name against category-specific patterns (case-insensitive)
4. For "Other": count drugs that don't match ANY category
"""
