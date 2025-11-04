"""
Fix prescriptions table to add admission_id column mapping to hadm_id
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
print("Fix Prescriptions Table - Add admission_id Column")
print("=" * 80)
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print()

conn = psycopg2.connect(**db_config)
conn.autocommit = True
cursor = conn.cursor()

# Check if admission_id column already exists
cursor.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name='prescriptions' AND column_name='admission_id'
""")
exists = cursor.fetchone()

if exists:
    print("✅ admission_id column already exists!")
else:
    print("Adding admission_id column...")
    cursor.execute("""
        ALTER TABLE prescriptions 
        ADD COLUMN admission_id VARCHAR(50)
    """)
    print("✅ Column added!")
    
    print("Populating admission_id from hadm_id...")
    cursor.execute("""
        UPDATE prescriptions 
        SET admission_id = hadm_id
    """)
    rows_updated = cursor.rowcount
    print(f"✅ Updated {rows_updated} rows!")
    
    print("Creating index on admission_id...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_prescriptions_admission_id 
        ON prescriptions(admission_id)
    """)
    print("✅ Index created!")

# Verify
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(admission_id) as with_admission_id,
        COUNT(hadm_id) as with_hadm_id,
        COUNT(DISTINCT admission_id) as unique_admissions
    FROM prescriptions
""")
total, with_adm, with_hadm, unique_adm = cursor.fetchone()
print()
print("Verification:")
print(f"  Total rows: {total}")
print(f"  Rows with admission_id: {with_adm}")
print(f"  Rows with hadm_id: {with_hadm}")
print(f"  Unique admissions: {unique_adm}")

# Sample data
cursor.execute("""
    SELECT prescription_id, patient_id, hadm_id, admission_id, drug 
    FROM prescriptions 
    LIMIT 3
""")
print("\nSample data:")
for row in cursor.fetchall():
    print(f"  ID:{row[0]} | patient:{row[1]} | hadm_id:{row[2]} | admission_id:{row[3]} | drug:{row[4]}")

cursor.close()
conn.close()

print()
print("=" * 80)
print("✅ Prescriptions table fixed!")
print("=" * 80)
print()
print("Now prescriptions table has:")
print("  - subject_id (MIMIC-IV original)")
print("  - patient_id (for feature calculator)")
print("  - hadm_id (MIMIC-IV original)")
print("  - admission_id (for feature calculator)")
