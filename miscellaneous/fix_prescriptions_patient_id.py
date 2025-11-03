"""
Fix prescriptions table to add patient_id column mapping to subject_id
This maintains MIMIC-IV compatibility while supporting feature calculator
"""
import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
import sys
sys.path.append('..')
load_dotenv('api/.env')

db_config = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

print("=" * 80)
print("Fix Prescriptions Table - Add patient_id Column")
print("=" * 80)
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print()

conn = psycopg2.connect(**db_config)
conn.autocommit = True
cursor = conn.cursor()

# Check if patient_id column already exists
cursor.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name='prescriptions' AND column_name='patient_id'
""")
exists = cursor.fetchone()

if exists:
    print("✅ patient_id column already exists!")
else:
    print("Adding patient_id column...")
    cursor.execute("""
        ALTER TABLE prescriptions 
        ADD COLUMN patient_id VARCHAR(50)
    """)
    print("✅ Column added!")
    
    print("Populating patient_id from subject_id...")
    cursor.execute("""
        UPDATE prescriptions 
        SET patient_id = subject_id
    """)
    rows_updated = cursor.rowcount
    print(f"✅ Updated {rows_updated} rows!")
    
    print("Creating index on patient_id...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prescriptions_patient_id 
        ON prescriptions(patient_id)
    """)
    print("✅ Index created!")

# Verify
cursor.execute("""
    SELECT COUNT(*), COUNT(DISTINCT patient_id), COUNT(DISTINCT subject_id)
    FROM prescriptions
""")
total, unique_patients, unique_subjects = cursor.fetchone()
print()
print("Verification:")
print(f"  Total rows: {total}")
print(f"  Unique patient_id values: {unique_patients}")
print(f"  Unique subject_id values: {unique_subjects}")

# Sample data
cursor.execute("""
    SELECT prescription_id, subject_id, patient_id, drug, drug_name, generic_name 
    FROM prescriptions 
    LIMIT 3
""")
print("\nSample data:")
for row in cursor.fetchall():
    print(f"  ID:{row[0]} | subject_id:{row[1]} | patient_id:{row[2]} | drug:{row[3]}")

cursor.close()
conn.close()

print()
print("=" * 80)
print("✅ Prescriptions table fixed!")
print("=" * 80)
print()
print("Now prescriptions table has both:")
print("  - subject_id (MIMIC-IV original)")
print("  - patient_id (for feature calculator compatibility)")
