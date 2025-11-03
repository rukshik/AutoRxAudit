-- AutoRxAudit Database Schema
-- Patient features table for opioid prescription auditing

CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    
    -- Demographics
    age_at_first_admit FLOAT NOT NULL,
    
    -- BMI & Obesity
    bmi FLOAT,
    has_bmi BOOLEAN DEFAULT FALSE,
    obesity_flag BOOLEAN DEFAULT FALSE,
    
    -- Hospital Admission Statistics
    n_hospital_admits INTEGER DEFAULT 0,
    n_admissions_with_drg INTEGER DEFAULT 0,
    n_icu_admissions INTEGER DEFAULT 0,
    n_icu_stays INTEGER DEFAULT 0,
    
    -- Length of Stay
    total_los_days FLOAT DEFAULT 0,
    avg_los_days FLOAT DEFAULT 0,
    total_icu_days FLOAT DEFAULT 0,
    total_icu_hours FLOAT DEFAULT 0,
    
    -- DRG Severity Indicators
    avg_drg_severity FLOAT DEFAULT 0,
    max_drg_severity FLOAT DEFAULT 0,
    avg_drg_mortality FLOAT DEFAULT 0,
    high_severity_flag BOOLEAN DEFAULT FALSE,
    
    -- Medication History (ATC codes)
    atc_A_rx_count INTEGER DEFAULT 0,  -- Alimentary tract and metabolism
    atc_B_rx_count INTEGER DEFAULT 0,  -- Blood and blood forming organs
    atc_C_rx_count INTEGER DEFAULT 0,  -- Cardiovascular system
    
    -- Opioid Exposure (for OUD Risk model only)
    opioid_rx_count INTEGER DEFAULT 0,
    distinct_opioids INTEGER DEFAULT 0,
    opioid_exposure_days FLOAT DEFAULT 0,
    opioid_hadms INTEGER DEFAULT 0,
    any_opioid_flag BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_patient_id ON patients(patient_id);

-- Audit log table
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
    
    -- Reasons
    flag_reason TEXT,
    
    -- Metadata
    audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_date ON audit_logs(audited_at);
CREATE INDEX IF NOT EXISTS idx_audit_flagged ON audit_logs(flagged);

-- Comments
COMMENT ON TABLE patients IS 'Patient feature data for opioid prescription auditing';
COMMENT ON TABLE audit_logs IS 'Historical audit decisions and model predictions';
COMMENT ON COLUMN patients.opioid_rx_count IS 'Used ONLY for OUD Risk model, NOT for Eligibility model';
COMMENT ON COLUMN audit_logs.flagged IS 'True if prescription should be flagged for review';
