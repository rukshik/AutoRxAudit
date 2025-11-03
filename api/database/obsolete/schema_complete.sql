-- AutoRxAudit Database Schema - Complete Workflow Tracking
-- Extends schema_raw.sql with prescriptions, audit logs, and user actions

-- ============================================================================
-- USERS TABLE - Authentication
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'clinician',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);

-- ============================================================================
-- PRESCRIPTIONS TABLE - Prescription orders
-- ============================================================================
CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    drug_name VARCHAR(255) NOT NULL,
    prescriber_id INTEGER REFERENCES users(user_id),
    prescribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'PENDING', -- 'PENDING', 'APPROVED', 'DENIED'
    
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_prescriptions_patient ON prescriptions(patient_id);
CREATE INDEX idx_prescriptions_prescriber ON prescriptions(prescriber_id);
CREATE INDEX idx_prescriptions_status ON prescriptions(status);
CREATE INDEX idx_prescriptions_created ON prescriptions(prescribed_at DESC);

-- ============================================================================
-- AUDIT LOGS TABLE - Complete audit trail (AI + Pharmacist decision)
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id SERIAL PRIMARY KEY,
    prescription_id INTEGER REFERENCES prescriptions(prescription_id),
    patient_id VARCHAR(50) NOT NULL,
    drug_name VARCHAR(255) NOT NULL,
    
    -- AI Model Results
    eligibility_score FLOAT NOT NULL,
    eligibility_prediction INTEGER NOT NULL,
    oud_risk_score FLOAT NOT NULL,
    oud_risk_prediction INTEGER NOT NULL,
    flagged BOOLEAN NOT NULL,
    flag_reason TEXT,
    recommendation TEXT,
    
    -- Pharmacist Decision (filled after review)
    reviewed_by INTEGER REFERENCES users(user_id),
    action VARCHAR(50), -- 'APPROVED', 'DENIED', 'OVERRIDE_APPROVE', 'OVERRIDE_DENY'
    action_reason TEXT,
    reviewed_at TIMESTAMP,
    
    -- Timestamps
    audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_audit_logs_prescription ON audit_logs(prescription_id);
CREATE INDEX idx_audit_logs_patient ON audit_logs(patient_id);
CREATE INDEX idx_audit_logs_flagged ON audit_logs(flagged);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_reviewed_by ON audit_logs(reviewed_by);
CREATE INDEX idx_audit_logs_audited ON audit_logs(audited_at DESC);
CREATE INDEX idx_audit_logs_reviewed ON audit_logs(reviewed_at DESC);

-- ============================================================================
-- SAMPLE USERS (for development)
-- ============================================================================
-- Plain text passwords for development only!

INSERT INTO users (email, password, full_name, role) VALUES 
    ('doctor@hospital.com', 'password123', 'Dr. Sarah Johnson', 'doctor'),
    ('pharmacist@hospital.com', 'password123', 'John Smith RPh', 'pharmacist'),
    ('admin@hospital.com', 'admin123', 'Admin User', 'admin')
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE users IS 'User accounts for authentication';
COMMENT ON TABLE prescriptions IS 'Prescription orders submitted for audit';
COMMENT ON TABLE audit_logs IS 'Complete audit trail: AI results + Pharmacist decision';

COMMENT ON COLUMN prescriptions.status IS 'PENDING (awaiting decision), APPROVED (cleared), DENIED (rejected)';
COMMENT ON COLUMN audit_logs.flagged IS 'TRUE if AI flagged prescription for review';
COMMENT ON COLUMN audit_logs.action IS 'Pharmacist decision: APPROVED, DENIED, OVERRIDE_APPROVE, OVERRIDE_DENY';
COMMENT ON COLUMN audit_logs.reviewed_by IS 'User ID of pharmacist who reviewed';
COMMENT ON COLUMN audit_logs.reviewed_at IS 'Timestamp when pharmacist made decision';
