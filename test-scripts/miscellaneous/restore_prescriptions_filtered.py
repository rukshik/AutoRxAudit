"""
Restore MIMIC-IV prescriptions table - FILTERED for 500 patients only
Only loads prescriptions for patients that exist in the patients table
"""
import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd
import gzip
from psycopg2.extras import execute_values

# Load environment variables from the correct path
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
print("Restore MIMIC-IV Prescriptions Table - FILTERED FOR 500 PATIENTS")
print("=" * 80)
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print()

# Connect to database
print("Connecting to database...")
conn = psycopg2.connect(**db_config)
conn.autocommit = True
cursor = conn.cursor()
print("✅ Connected!")
print()

# Get list of patient IDs from patients table
print("Fetching patient IDs from patients table...")
cursor.execute("SELECT patient_id FROM patients ORDER BY patient_id")
patient_ids = [row[0] for row in cursor.fetchall()]
print(f"✅ Found {len(patient_ids)} patients in patients table")
print(f"Patient ID range: {patient_ids[0]} to {patient_ids[-1]}")
print()

# Drop existing table if it exists
print("Dropping existing prescriptions table if present...")
cursor.execute("DROP TABLE IF EXISTS prescriptions CASCADE;")
print("✅ Dropped!")

# Create prescriptions table matching MIMIC-IV demo CSV structure
print("Creating MIMIC-IV prescriptions table...")
cursor.execute("""
    CREATE TABLE prescriptions (
        prescription_id SERIAL PRIMARY KEY,
        subject_id VARCHAR(50) NOT NULL,
        hadm_id VARCHAR(50),
        pharmacy_id VARCHAR(50),
        starttime TIMESTAMP,
        stoptime TIMESTAMP,
        drug_type VARCHAR(50),
        drug TEXT,
        drug_name TEXT,
        generic_name TEXT,
        formulary_drug_cd VARCHAR(50),
        gsn TEXT,
        ndc VARCHAR(50),
        prod_strength TEXT,
        form_rx VARCHAR(50),
        dose_val_rx VARCHAR(100),
        dose_unit_rx VARCHAR(50),
        form_val_disp VARCHAR(100),
        form_unit_disp VARCHAR(50),
        doses_per_24_hrs FLOAT,
        route VARCHAR(50),
        
        -- Compatibility columns for feature calculator
        patient_id VARCHAR(50),
        admission_id VARCHAR(50),
        start_time TIMESTAMP,
        stop_time TIMESTAMP
    );
""")
print("✅ Table created!")

# Create indexes
print("Creating indexes...")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_subject ON prescriptions(subject_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_patient ON prescriptions(patient_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_hadm ON prescriptions(hadm_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_admission ON prescriptions(admission_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_drug ON prescriptions(drug);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_drug_name ON prescriptions(drug_name);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_generic ON prescriptions(generic_name);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_dates ON prescriptions(starttime, stoptime);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_dates_compat ON prescriptions(start_time, stop_time);")
print("✅ Indexes created!")
print()

# Load data from CSV - USE SYNTHETIC DATASET (matches patient table)
csv_path = 'datasets/synthetic_mimic_50000_v3/mimic-clinical-iv-demo/hosp/prescriptions.csv.gz'
print(f"Loading data from {csv_path}...")
print(f"Filtering for {len(patient_ids)} patients only...")

try:
    # Read CSV in chunks to handle large file efficiently
    chunk_size = 100000
    total_loaded = 0
    total_inserted = 0
    
    # Convert patient_ids to set for faster lookup
    patient_id_set = set(patient_ids)
    
    for chunk_num, df_chunk in enumerate(pd.read_csv(csv_path, compression='gzip', chunksize=chunk_size), 1):
        print(f"Processing chunk {chunk_num} ({len(df_chunk)} rows)...")
        
        # Filter for our 500 patients only
        df_filtered = df_chunk[df_chunk['subject_id'].astype(str).isin(patient_id_set)]
        
        if len(df_filtered) == 0:
            print(f"  No matching patients in this chunk, skipping...")
            total_loaded += len(df_chunk)
            continue
        
        print(f"  Found {len(df_filtered)} prescriptions for our patients")
        
        # Replace NaN with None for proper NULL insertion
        df_filtered = df_filtered.where(pd.notna(df_filtered), None)
        
        # Prepare data for insertion with compatibility columns
        data_to_insert = []
        for _, row in df_filtered.iterrows():
            data_to_insert.append((
                str(row['subject_id']),
                str(row['hadm_id']) if row['hadm_id'] is not None else None,
                str(row['pharmacy_id']) if row['pharmacy_id'] is not None else None,
                row['starttime'],
                row['stoptime'],
                row['drug_type'],
                row['drug'],
                row['drug'],  # drug_name = drug
                row['prod_strength'],  # generic_name = prod_strength
                row['formulary_drug_cd'],
                row['gsn'],
                row['ndc'],
                row['prod_strength'],
                row['form_rx'],
                row['dose_val_rx'],
                row['dose_unit_rx'],
                row['form_val_disp'],
                row['form_unit_disp'],
                row['doses_per_24_hrs'],
                row['route'],
                # Compatibility columns
                str(row['subject_id']),  # patient_id = subject_id
                str(row['hadm_id']) if row['hadm_id'] is not None else None,  # admission_id = hadm_id
                row['starttime'],  # start_time = starttime
                row['stoptime']  # stop_time = stoptime
            ))
        
        # Batch insert
        execute_values(
            cursor,
            """
            INSERT INTO prescriptions (
                subject_id, hadm_id, pharmacy_id, starttime, stoptime,
                drug_type, drug, drug_name, generic_name, formulary_drug_cd,
                gsn, ndc, prod_strength, form_rx, dose_val_rx, dose_unit_rx,
                form_val_disp, form_unit_disp, doses_per_24_hrs, route,
                patient_id, admission_id, start_time, stop_time
            ) VALUES %s
            """,
            data_to_insert
        )
        
        total_loaded += len(df_chunk)
        total_inserted += len(df_filtered)
        print(f"  ✅ Inserted {len(df_filtered)} rows (total: {total_inserted})")
    
    print()
    print(f"✅ Processed {total_loaded} total rows from CSV")
    print(f"✅ Inserted {total_inserted} prescriptions for {len(patient_ids)} patients")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    cursor.close()
    conn.close()
    exit(1)

# Verify data
print()
print("Verifying data...")
cursor.execute("SELECT COUNT(*) FROM prescriptions")
total_count = cursor.fetchone()[0]
print(f"Total rows in prescriptions table: {total_count}")

cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM prescriptions")
unique_patients = cursor.fetchone()[0]
print(f"Unique patients: {unique_patients}")

cursor.execute("""
    SELECT patient_id, drug_name, generic_name, start_time 
    FROM prescriptions 
    ORDER BY patient_id, start_time 
    LIMIT 5
""")
samples = cursor.fetchall()
print()
print("Sample data from database:")
for row in samples:
    print(f"  Patient: {row[0]}, Drug: {row[1]}, Generic: {row[2]}, Start: {row[3]}")

# Check specific patient 20000199
print()
print("Checking patient 20000199...")
cursor.execute("""
    SELECT COUNT(*) 
    FROM prescriptions 
    WHERE patient_id = '20000199'
""")
patient_rx_count = cursor.fetchone()[0]
print(f"Patient 20000199 has {patient_rx_count} total prescriptions")

# Check opioid prescriptions for patient 20000199
cursor.execute("""
    SELECT COUNT(*) 
    FROM prescriptions 
    WHERE patient_id = '20000199'
    AND (
        drug_name ~* 'fentanyl|morphine|oxycodone|hydrocodone|hydromorphone|oxymorphone|codeine|tramadol|methadone|buprenorphine|meperidine|tapentadol'
        OR generic_name ~* 'fentanyl|morphine|oxycodone|hydrocodone|hydromorphone|oxymorphone|codeine|tramadol|methadone|buprenorphine|meperidine|tapentadol'
    )
""")
opioid_rx_count = cursor.fetchone()[0]
print(f"Patient 20000199 has {opioid_rx_count} opioid prescriptions")

cursor.close()
conn.close()

print()
print("=" * 80)
print("✅ MIMIC-IV prescriptions table restored (FILTERED FOR 500 PATIENTS)!")
print("=" * 80)
print()
print("Note: This is separate from prescription_requests table (workflow)")
print("  - prescriptions: MIMIC-IV historical EHR data for feature calculation")
print("  - prescription_requests: Workflow table for new prescription submissions")
