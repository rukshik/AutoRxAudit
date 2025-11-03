-- Pharmacy Database Schema

-- Patients Table (ID is same in doctor and pharmacy systems)
CREATE TABLE patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users Table (pharmacists, pharmacy staff)
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('pharmacist', 'pharmacy_tech', 'admin', 'staff')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Prescription Requests Table (pharmacy side)
CREATE TABLE prescription_requests (
    id SERIAL PRIMARY KEY,
    prescription_uuid UUID UNIQUE NOT NULL,
    patient_id VARCHAR(50) NOT NULL REFERENCES patients(patient_id),
    prescribing_doctor_name VARCHAR(200) NOT NULL,
    prescribing_doctor_id VARCHAR(100),
    drug_name VARCHAR(200) NOT NULL,
    dosage VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    refills INTEGER DEFAULT 0,
    instructions TEXT,
    diagnosis VARCHAR(500),
    
    -- AI Analysis Fields
    ai_status VARCHAR(50) CHECK (ai_status IN ('pending', 'analyzing', 'flagged', 'approved', 'error')),
    ai_eligibility_score DECIMAL(5,2),
    ai_oud_risk_score DECIMAL(5,2),
    ai_flag_reason TEXT,
    ai_recommendation TEXT,
    ai_analyzed_at TIMESTAMP,
    
    -- Pharmacist Decision Fields
    pharmacist_decision VARCHAR(50) CHECK (pharmacist_decision IN ('pending', 'approved', 'denied', 'needs_clarification')),
    decision_made_by INTEGER REFERENCES users(user_id),
    decision_reason TEXT,
    decision_time TIMESTAMP,
    
    status VARCHAR(50) DEFAULT 'received' CHECK (status IN ('received', 'under_review', 'approved', 'denied', 'dispensed', 'cancelled')),
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables
CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prescription_requests_updated_at BEFORE UPDATE ON prescription_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
