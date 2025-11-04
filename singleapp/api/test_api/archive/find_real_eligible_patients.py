import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cursor = conn.cursor()

# Find patients that exist in patients table AND have pain diagnoses
cursor.execute("""
    SELECT DISTINCT p.patient_id
    FROM patients p
    JOIN diagnoses d ON p.patient_id = d.patient_id
    WHERE d.icd_code LIKE 'M%'    -- ICD-10 musculoskeletal/pain
       OR d.icd_code LIKE '338%'  -- ICD-9 pain
       OR d.icd_code LIKE '72%'   -- ICD-9 back pain
       OR d.icd_code LIKE 'G89%'  -- ICD-10 chronic pain
    LIMIT 10
""")

eligible_patients = [row[0] for row in cursor.fetchall()]

print(f"Found {len(eligible_patients)} eligible patients (exist in patients table + have pain diagnoses):")
print(eligible_patients)

# Also check: patients with opioid prescriptions
cursor.execute("""
    SELECT DISTINCT p.patient_id
    FROM patients p
    JOIN prescriptions pr ON p.patient_id = pr.patient_id
    WHERE pr.drug_name ILIKE '%oxycodone%'
       OR pr.drug_name ILIKE '%morphine%'
    LIMIT 10
""")

opioid_patients = [row[0] for row in cursor.fetchall()]

print(f"\nFound {len(opioid_patients)} patients with opioid history:")
print(opioid_patients)

cursor.close()
conn.close()
