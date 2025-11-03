# Database Setup Instructions

## Overview
This system uses two separate databases on the same PostgreSQL server:
- **doctor_office**: Doctor's office prescription management
- **pharmacy**: Pharmacy prescription processing and dispensing

Both databases share the same patient IDs but maintain separate prescription records linked by `prescription_uuid`.

## Database Connection Details
- **Host**: rxaudit.postgres.database.azure.com
- **Port**: 5432
- **User**: posgres
- **Password**: UmaKiran12

## Setup Steps

### 1. Create Doctor Office Database

```bash
# Connect to PostgreSQL
psql -h rxaudit.postgres.database.azure.com -U posgres -d postgres

# Run the creation script
\i doctor/database/01_create_doctor_db.sql

# Connect to the new database
\c doctor_office

# Create schema
\i doctor/database/02_schema.sql

# Insert seed data
\i doctor/database/03_seed_data.sql
```

### 2. Create Pharmacy Database

```bash
# Connect to PostgreSQL (if not already connected)
psql -h rxaudit.postgres.database.azure.com -U posgres -d postgres

# Run the creation script
\i pharmacy/database/01_create_pharmacy_db.sql

# Connect to the new database
\c pharmacy

# Create schema
\i pharmacy/database/02_schema.sql

# Insert seed data
\i pharmacy/database/03_seed_data.sql
```

## Database Schemas

### Doctor Office Database

**Tables:**
- `patients`: Shared patient records (P001-P010)
- `users`: Doctor office staff (doctors, physician assistants, admin)
- `prescription_requests`: Prescriptions created by doctors
  - Includes `pharmacy_decision` column to track pharmacy response
- `prescription_history`: Audit trail of prescription changes

**Sample Users:**
- `jdoctor` / `password123` - Dr. Joe Doctor (doctor)
- `mpa` / `password123` - Mary Physician Assistant (physician_assistant)
- `sadmin` / `password123` - Sarah Admin (admin)
- `bnurse` / `password123` - Bob Nurse (staff)

### Pharmacy Database

**Tables:**
- `patients`: Shared patient records (same IDs as doctor system)
- `users`: Pharmacy staff (pharmacists, pharmacy techs, admin)
- `prescription_requests`: Prescriptions received from doctors
  - Includes AI analysis fields: `ai_status`, `ai_eligibility_score`, `ai_oud_risk_score`
  - Includes pharmacist decision fields: `pharmacist_decision`, `decision_made_by`, `decision_time`
- `prescription_history`: Audit trail of prescription processing
- `dispensing_records`: Record of actual medication dispensing

**Sample Users:**
- `spharma` / `password123` - Susan Pharmacist (pharmacist, License: RPH12345)
- `badmin` / `password123` - Bob Pharmacy Admin (admin)
- `jtech` / `password123` - John Pharmacy Tech (pharmacy_tech)
- `lpharmacist` / `password123` - Lisa Pharmacist (pharmacist, License: RPH54321)

## Sample Patients (Shared across both systems)

| Patient ID | Name | DOB | Phone |
|------------|------|-----|-------|
| P001 | Joe Doe | 1980-05-15 | 555-0101 |
| P002 | Jane Doe | 1975-08-22 | 555-0102 |
| P003 | Mary Smith | 1990-03-10 | 555-0103 |
| P004 | Bob Johnson | 1965-11-30 | 555-0104 |
| P005 | Alice Brown | 1988-07-18 | 555-0105 |
| P006 | Tom Taylor | 1972-09-25 | 555-0106 |
| P007 | Susan Jones | 1995-12-05 | 555-0107 |
| P008 | David Wilson | 1982-04-14 | 555-0108 |
| P009 | Emily Davis | 1978-06-20 | 555-0109 |
| P010 | Michael Miller | 1968-02-28 | 555-0110 |

## Prescription Workflow

1. **Doctor creates prescription** in `doctor_office.prescription_requests`
   - Generates unique `prescription_uuid`
   - Status: `sent_to_pharmacy`
   - `pharmacy_decision`: `pending`

2. **Prescription sent to pharmacy** (via API)
   - Inserted into `pharmacy.prescription_requests` with same `prescription_uuid`
   - Status: `received`
   - AI analysis runs automatically

3. **Pharmacy processes prescription**
   - AI sets `ai_status`, `ai_eligibility_score`, `ai_oud_risk_score`
   - Pharmacist reviews and sets `pharmacist_decision`
   - Updates `decision_made_by`, `decision_time`

4. **Pharmacy sends decision back to doctor** (via API)
   - Updates `doctor_office.prescription_requests.pharmacy_decision`
   - Updates `pharmacy_decision_time`, `pharmacy_decision_notes`

## Verification Queries

```sql
-- Doctor Office: Check prescriptions
SELECT 
    prescription_uuid,
    p.first_name || ' ' || p.last_name as patient_name,
    drug_name,
    status,
    pharmacy_decision
FROM doctor_office.prescription_requests pr
JOIN doctor_office.patients p ON pr.patient_id = p.patient_id
ORDER BY created_at DESC;

-- Pharmacy: Check prescriptions with AI analysis
SELECT 
    prescription_uuid,
    p.first_name || ' ' || p.last_name as patient_name,
    drug_name,
    ai_status,
    ai_oud_risk_score,
    pharmacist_decision,
    status
FROM pharmacy.prescription_requests pr
JOIN pharmacy.patients p ON pr.patient_id = p.patient_id
ORDER BY received_at DESC;
```

## Next Steps

After database setup:
1. Create Doctor Office API (Port 8003)
2. Create Pharmacy API (Port 8004)
3. Create Doctor Frontend App (Port 3001)
4. Create Pharmacy Frontend App (Port 3002)
