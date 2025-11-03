-- Seed Data for Pharmacy Database

-- Insert Pharmacy Users
INSERT INTO users (password, email, full_name, role, is_active) VALUES
('password123', 'susan.pharmacist@pharmacy.com', 'Susan Pharmacist', 'pharmacist', TRUE),
('password123', 'bob.admin@pharmacy.com', 'Bob Pharmacy Admin', 'admin', TRUE),
('password123', 'john.tech@pharmacy.com', 'John Pharmacy Tech', 'pharmacy_tech', TRUE),
('password123', 'lisa.pharmacist@pharmacy.com', 'Lisa Pharmacist', 'pharmacist', TRUE);

-- Insert Patients (500 patients from autorxaudit database with auto-generated unique names)
-- Same patients as doctor database for consistency
-- Data source: patients_data.sql
\i patients_data.sql

