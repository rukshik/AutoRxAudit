-- Doctor Office Database Schema

-- Patients Table (ID is same in doctor and pharmacy systems)
CREATE TABLE patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Doctor's office users
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('doctor', 'physician_assistant', 'admin', 'staff')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);


-- Prescriptions Table (doctor side)
CREATE TABLE prescription_requests (
    id SERIAL PRIMARY KEY,
    prescription_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) NOT NULL REFERENCES patients(patient_id),
    prescriber_id INTEGER NOT NULL REFERENCES users(user_id),
    drug_name VARCHAR(200) NOT NULL,
    dosage VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    refills INTEGER DEFAULT 0,
    pharmacy_notes VARCHAR(3000),
    status VARCHAR(50) DEFAULT 'sent_to_pharmacy' CHECK (status IN ('draft', 'sent_to_pharmacy', 'pharmacy_approved', 'pharmacy_denied', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
