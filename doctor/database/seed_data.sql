-- Seed Data for Doctor Office Database

-- Insert Doctor Office Users
INSERT INTO users (password, email, full_name, role, is_active) VALUES
('password123', 'joe.doctor@clinic.com', 'Dr. Joe Doctor', 'doctor', TRUE),
('password123', 'mary.pa@clinic.com', 'Mary Physician Assistant', 'physician_assistant', TRUE),
('password123', 'sarah.admin@clinic.com', 'Sarah Admin', 'admin', TRUE),
('password123', 'bob.staff@clinic.com', 'Bob Office Staff', 'staff', TRUE);

-- Insert Patients (500 patients from autorxaudit database with auto-generated unique names)
-- Data source: patients_data.sql
\i patients_data.sql


