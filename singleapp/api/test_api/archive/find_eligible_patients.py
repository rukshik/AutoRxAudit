import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cursor = conn.cursor()

# Find patients with pain diagnoses (likely eligible)
print("Finding patients with pain diagnoses...")
cursor.execute("""
    SELECT DISTINCT d.patient_id
    FROM diagnoses d
    WHERE d.icd_code LIKE 'M%'    -- ICD-10 musculoskeletal/pain
       OR d.icd_code LIKE '338%'  -- ICD-9 pain
       OR d.icd_code LIKE '72%'   -- ICD-9 back pain
       OR d.icd_code LIKE 'G89%'  -- ICD-10 chronic pain
    LIMIT 20
""")

pain_patients = [row[0] for row in cursor.fetchall()]

print(f"\nFound {len(pain_patients)} patients with pain diagnoses:")
print(pain_patients[:10])

# Also find patients who received opioid prescriptions
print("\n\nFinding patients who were prescribed opioids...")
cursor.execute("""
    SELECT DISTINCT patient_id
    FROM prescriptions
    WHERE drug_name ILIKE '%oxycodone%'
       OR drug_name ILIKE '%morphine%'
       OR drug_name ILIKE '%hydrocodone%'
       OR drug_name ILIKE '%fentanyl%'
    LIMIT 20
""")

opioid_patients = [row[0] for row in cursor.fetchall()]

print(f"\nFound {len(opioid_patients)} patients who received opioids:")
print(opioid_patients[:10])

# Best candidates: pain diagnosis AND received opioids (historically eligible)
print("\n\nFinding best candidates (pain + opioid history)...")
cursor.execute("""
    SELECT DISTINCT d.patient_id
    FROM diagnoses d
    JOIN prescriptions p ON d.patient_id = p.patient_id
    WHERE (d.icd_code LIKE 'M%' OR d.icd_code LIKE '338%' OR d.icd_code LIKE '72%')
      AND (p.drug_name ILIKE '%oxycodone%' OR p.drug_name ILIKE '%morphine%')
    LIMIT 10
""")

best_candidates = [row[0] for row in cursor.fetchall()]

print(f"\nFound {len(best_candidates)} ideal test patients:")
print(best_candidates)

cursor.close()
conn.close()
