"""
Restore MIMIC-IV prescriptions table that was accidentally dropped
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
print("Restore MIMIC-IV Prescriptions Table")
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
        route VARCHAR(50)
    );
""")
print("✅ Table created!")

# Create indexes
print("Creating indexes...")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_subject ON prescriptions(subject_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_hadm ON prescriptions(hadm_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_drug ON prescriptions(drug);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_drug_name ON prescriptions(drug_name);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_generic ON prescriptions(generic_name);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_dates ON prescriptions(starttime, stoptime);")
print("✅ Indexes created!")
print()

# Load data from CSV - USE SYNTHETIC DATASET (matches patient table)
csv_path = 'datasets/synthetic_mimic_50000_v3/mimic-clinical-iv-demo/hosp/prescriptions.csv.gz'
print(f"Loading data from {csv_path}...")

try:
    # Read CSV
    df = pd.read_csv(csv_path, compression='gzip')
    print(f"✅ Loaded {len(df)} rows from CSV")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Map CSV columns to database columns (MIMIC-IV uses different column names)
    # subject_id -> patient_id
    # hadm_id -> admission_id
    # drug -> drug_name
    # (other mappings as needed)
    
    # Check what columns we have
    print("Sample data:")
    print(df.head(2))
    print()
    
    # Prepare data for insertion
    print("Inserting data into database...")
    
    # Replace NaN with None for proper NULL insertion
    df = df.where(pd.notna(df), None)
    
    # Convert dataframe to list of tuples matching CSV columns
    data_to_insert = []
    for _, row in df.iterrows():
        # Also populate drug_name and generic_name for compatibility with feature calculator
        data_to_insert.append((
            str(row['subject_id']),
            str(row['hadm_id']) if row['hadm_id'] is not None else None,
            str(row['pharmacy_id']) if row['pharmacy_id'] is not None else None,
            row['starttime'],
            row['stoptime'],
            row['drug_type'],
            row['drug'],
            row['drug'],  # drug_name = drug
            row['prod_strength'],  # generic_name = prod_strength (closest approximation)
            row['formulary_drug_cd'],
            row['gsn'],
            row['ndc'],
            row['prod_strength'],
            row['form_rx'],
            row['dose_val_rx'],
            row['dose_unit_rx'],
            row['form_val_disp'],
            row['form_unit_disp'],
            float(row['doses_per_24_hrs']) if row['doses_per_24_hrs'] is not None else None,
            row['route']
        ))
    
    # Batch insert using execute_values for efficiency
    execute_values(
        cursor,
        """
        INSERT INTO prescriptions (
            subject_id, hadm_id, pharmacy_id, starttime, stoptime,
            drug_type, drug, drug_name, generic_name, formulary_drug_cd,
            gsn, ndc, prod_strength, form_rx, dose_val_rx, dose_unit_rx,
            form_val_disp, form_unit_disp, doses_per_24_hrs, route
        ) VALUES %s
        """,
        data_to_insert,
        page_size=1000
    )
    
    print(f"✅ Inserted {len(data_to_insert)} rows")
    print()
    
    # Verify insertion
    cursor.execute("SELECT COUNT(*) FROM prescriptions")
    count = cursor.fetchone()[0]
    print(f"Total rows in prescriptions table: {count}")
    
    # Show sample
    cursor.execute("SELECT subject_id, drug, drug_name, generic_name, starttime FROM prescriptions LIMIT 3")
    print("\nSample data from database:")
    for row in cursor.fetchall():
        print(f"  Patient: {row[0]}, Drug: {row[1]}, DrugName: {row[2]}, Generic: {row[3]}, Start: {row[4]}")
    
except FileNotFoundError:
    print(f"❌ Error: CSV file not found at {csv_path}")
    print("Please check the file path and try again.")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()

cursor.close()
conn.close()

print()
print("=" * 80)
print("✅ MIMIC-IV prescriptions table restored!")
print("=" * 80)
print()
print("Note: This is separate from prescription_requests table (workflow)")
print("  - prescriptions: MIMIC-IV historical EHR data for feature calculation")
print("  - prescription_requests: Workflow table for new prescription submissions")
