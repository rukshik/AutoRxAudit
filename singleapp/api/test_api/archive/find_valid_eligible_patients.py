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

# Get sample patients from patients table
cursor.execute("SELECT patient_id FROM patients LIMIT 50")
all_patients = [row[0] for row in cursor.fetchall()]

print(f"Testing {len(all_patients)} patients for pain diagnoses...")

eligible_patients = []
for patient_id in all_patients:
    cursor.execute("""
        SELECT COUNT(*) 
        FROM diagnoses 
        WHERE patient_id = %s 
          AND (icd_code LIKE 'M%%' 
               OR icd_code LIKE '338%%' 
               OR icd_code LIKE '72%%'
               OR icd_code LIKE 'G89%%')
    """, (patient_id,))
    
    count = cursor.fetchone()[0]
    if count > 0:
        eligible_patients.append(patient_id)
        print(f"  ✓ {patient_id} has {count} pain diagnoses")
        
        if len(eligible_patients) >= 10:
            break

print(f"\nFound {len(eligible_patients)} eligible patients:")
print(eligible_patients)

cursor.close()
conn.close()
