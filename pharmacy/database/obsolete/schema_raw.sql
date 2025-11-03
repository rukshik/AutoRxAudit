-- AutoRxAudit Database Schema - EHR-like Raw Patient Data
-- Stores raw clinical data; API calculates features dynamically on request

-- ============================================================================
-- REFERENCE TABLES - Code lookups
-- ============================================================================

-- ICD-10 Code Reference
CREATE TABLE IF NOT EXISTS ref_icd_codes (
    icd_code VARCHAR(20) PRIMARY KEY,
    description TEXT,
    category VARCHAR(100),
    is_pain_related BOOLEAN DEFAULT FALSE,
    is_oud_related BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_icd_category ON ref_icd_codes(category);

-- ATC Drug Classification Reference
CREATE TABLE IF NOT EXISTS ref_atc_codes (
    atc_code VARCHAR(20) PRIMARY KEY,
    atc_level INTEGER, -- 1-5 (anatomical group to chemical substance)
    description TEXT,
    parent_code VARCHAR(20),
    is_opioid BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_atc_level ON ref_atc_codes(atc_level);
CREATE INDEX idx_atc_parent ON ref_atc_codes(parent_code);

-- Opioid Drug Reference
CREATE TABLE IF NOT EXISTS ref_opioid_drugs (
    drug_name VARCHAR(255) PRIMARY KEY,
    generic_name VARCHAR(255),
    atc_code VARCHAR(20),
    strength_unit VARCHAR(50),
    is_long_acting BOOLEAN DEFAULT FALSE,
    mme_conversion_factor FLOAT -- Morphine Milligram Equivalent
);

-- DRG Code Reference
CREATE TABLE IF NOT EXISTS ref_drg_codes (
    drg_code VARCHAR(20) PRIMARY KEY,
    description TEXT,
    mdc_code VARCHAR(10), -- Major Diagnostic Category
    severity_level INTEGER,
    mortality_risk INTEGER
);

-- ============================================================================
-- PATIENTS TABLE - Demographics
-- ============================================================================
CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10),
    race VARCHAR(50),
    ethnicity VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ADMISSIONS TABLE - Hospital admissions
-- ============================================================================
CREATE TABLE IF NOT EXISTS admissions (
    admission_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    admit_time TIMESTAMP NOT NULL,
    discharge_time TIMESTAMP,
    admission_type VARCHAR(50), -- 'ELECTIVE', 'EMERGENCY', 'URGENT'
    admission_location VARCHAR(100),
    discharge_location VARCHAR(100),
    insurance VARCHAR(50),
    language VARCHAR(50),
    marital_status VARCHAR(50),
    hospital_expire_flag BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_admissions_patient ON admissions(patient_id);
CREATE INDEX idx_admissions_dates ON admissions(admit_time, discharge_time);

-- ============================================================================
-- DIAGNOSES TABLE - ICD diagnosis codes
-- ============================================================================
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    admission_id VARCHAR(50),
    icd_code VARCHAR(20) NOT NULL,
    icd_version INTEGER DEFAULT 10,
    seq_num INTEGER, -- Diagnosis sequence (1=primary)
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id),
    FOREIGN KEY (icd_code) REFERENCES ref_icd_codes(icd_code)
);

CREATE INDEX idx_diagnoses_patient ON diagnoses(patient_id);
CREATE INDEX idx_diagnoses_admission ON diagnoses(admission_id);
CREATE INDEX idx_diagnoses_icd ON diagnoses(icd_code);
CREATE INDEX idx_diagnoses_primary ON diagnoses(seq_num) WHERE seq_num = 1;

-- ============================================================================
-- PRESCRIPTIONS TABLE - Medication orders
-- ============================================================================
CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    admission_id VARCHAR(50),
    drug_name VARCHAR(255) NOT NULL,
    generic_name VARCHAR(255),
    ndc_code VARCHAR(50), -- National Drug Code
    atc_code VARCHAR(20), -- ATC classification
    start_time TIMESTAMP NOT NULL,
    stop_time TIMESTAMP,
    dose_val_rx VARCHAR(100),
    dose_unit_rx VARCHAR(50),
    route VARCHAR(50), -- 'PO', 'IV', 'IM', etc.
    frequency VARCHAR(100),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
);

CREATE INDEX idx_prescriptions_patient ON prescriptions(patient_id);
CREATE INDEX idx_prescriptions_admission ON prescriptions(admission_id);
CREATE INDEX idx_prescriptions_drug ON prescriptions(drug_name);
CREATE INDEX idx_prescriptions_atc ON prescriptions(atc_code);
CREATE INDEX idx_prescriptions_dates ON prescriptions(start_time, stop_time);

-- ============================================================================
-- OMR (Observation Medical Record) - Vitals, BMI, etc.
-- ============================================================================
CREATE TABLE IF NOT EXISTS omr (
    omr_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    chart_time TIMESTAMP NOT NULL,
    result_name VARCHAR(100) NOT NULL, -- 'BMI', 'Weight', 'Height', 'BP Systolic', etc.
    result_type VARCHAR(50), -- 'Vital Sign', 'Body Measurement', 'Lab Value'
    result_value FLOAT,
    result_unit VARCHAR(20), -- 'kg', 'cm', 'mmHg', etc.
    normal_range_low FLOAT,
    normal_range_high FLOAT,
    is_abnormal BOOLEAN,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_omr_patient ON omr(patient_id);
CREATE INDEX idx_omr_result ON omr(result_name);
CREATE INDEX idx_omr_time ON omr(chart_time);
CREATE INDEX idx_omr_type ON omr(result_type);

-- ============================================================================
-- DRG CODES TABLE - Diagnosis Related Groups (severity indicators)
-- ============================================================================
CREATE TABLE IF NOT EXISTS drgcodes (
    drg_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    admission_id VARCHAR(50) NOT NULL,
    drg_code VARCHAR(20) NOT NULL,
    drg_type VARCHAR(50), -- 'MS' (Medicare Severity), 'APR' (All Patient Refined)
    drg_severity INTEGER, -- 1-4 scale
    drg_mortality INTEGER, -- 1-4 scale
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id),
    FOREIGN KEY (drg_code) REFERENCES ref_drg_codes(drg_code)
);

CREATE INDEX idx_drg_patient ON drgcodes(patient_id);
CREATE INDEX idx_drg_admission ON drgcodes(admission_id);
CREATE INDEX idx_drg_code ON drgcodes(drg_code);

-- ============================================================================
-- TRANSFERS TABLE - ICU stays and transfers
-- ============================================================================
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    admission_id VARCHAR(50) NOT NULL,
    in_time TIMESTAMP NOT NULL, -- Transfer into unit
    out_time TIMESTAMP, -- Transfer out of unit (NULL if still in unit)
    care_unit VARCHAR(100), -- 'MICU', 'SICU', 'CCU', 'Ward', 'Emergency', etc.
    careunit_type VARCHAR(50), -- 'ICU', 'Ward', 'Emergency', 'Operating Room'
    los_hours FLOAT, -- Length of stay in this unit (calculated)
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
);

CREATE INDEX idx_transfer_patient ON transfers(patient_id);
CREATE INDEX idx_transfer_admission ON transfers(admission_id);
CREATE INDEX idx_transfer_unit ON transfers(care_unit);
CREATE INDEX idx_transfer_type ON transfers(careunit_type);
CREATE INDEX idx_transfer_times ON transfers(in_time, out_time);

-- ============================================================================
-- AUDIT LOGS TABLE - AI predictions and decisions
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    prescription_id VARCHAR(50),
    
    -- Audit Results
    flagged BOOLEAN NOT NULL,
    eligibility_score FLOAT,
    eligibility_prediction INTEGER,
    oud_risk_score FLOAT,
    oud_risk_prediction INTEGER,
    
    -- Calculated Features (for debugging)
    calculated_features JSONB,
    
    -- Reasons
    flag_reason TEXT,
    
    -- Metadata
    audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_audit_patient ON audit_logs(patient_id);
CREATE INDEX idx_audit_date ON audit_logs(audited_at);
CREATE INDEX idx_audit_flagged ON audit_logs(flagged);

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE patients IS 'Patient demographics - raw EHR data';
COMMENT ON TABLE admissions IS 'Hospital admission records';
COMMENT ON TABLE diagnoses IS 'ICD diagnosis codes';
COMMENT ON TABLE prescriptions IS 'Medication orders and prescriptions';
COMMENT ON TABLE omr IS 'Vital signs and observation data (BMI, etc.)';
COMMENT ON TABLE drgcodes IS 'DRG severity and mortality indicators';
COMMENT ON TABLE transfers IS 'ICU admissions and unit transfers';
COMMENT ON TABLE audit_logs IS 'AI audit decisions with calculated features';
