-- AutoRxAudit Database Schema - With User Authentication and Audit Actions
-- Extends schema_raw.sql with user management and audit action tracking

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
-- AUDIT ACTIONS TABLE - User decisions on prescriptions
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_actions (
    action_id SERIAL PRIMARY KEY,
    audit_id INTEGER REFERENCES audit_logs(audit_id),
    user_id INTEGER REFERENCES users(user_id),
    patient_id VARCHAR(50) NOT NULL,
    drug_name VARCHAR(255) NOT NULL,
    
    -- AI Model Results
    flagged BOOLEAN NOT NULL,
    eligibility_score FLOAT,
    eligibility_prediction INTEGER,
    oud_risk_score FLOAT,
    oud_risk_prediction INTEGER,
    flag_reason TEXT,
    
    -- User Action
    action VARCHAR(50) NOT NULL, -- 'APPROVED', 'DENIED', 'OVERRIDE_APPROVE', 'OVERRIDE_DENY'
    action_reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_audit_actions_user ON audit_actions(user_id);
CREATE INDEX idx_audit_actions_patient ON audit_actions(patient_id);
CREATE INDEX idx_audit_actions_created ON audit_actions(created_at DESC);
CREATE INDEX idx_audit_actions_action ON audit_actions(action);

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
COMMENT ON TABLE audit_actions IS 'User actions on prescription audits (approve/deny/override)';
COMMENT ON COLUMN audit_actions.action IS 'User decision: APPROVED, DENIED, OVERRIDE_APPROVE, OVERRIDE_DENY';
