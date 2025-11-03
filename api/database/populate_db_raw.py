"""
Load 500 patients from synthetic MIMIC-IV data into raw EHR schema.
Loads from original CSV files: patients, admissions, diagnoses, prescriptions, omr, drgcodes, transfers.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime
import gzip
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'autorxaudit'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

# Data paths
DATA_DIR = "../../datasets/synthetic_mimic_50000_v3/mimic-clinical-iv-demo/hosp"  # Patient data
DEMO_DATA_DIR = "../../data/mimic-clinical-iv-demo/hosp"  # Reference data

def load_csv_gz(filename, data_dir=None):
    """Load a gzipped CSV file."""
    if data_dir is None:
        data_dir = DATA_DIR
    filepath = os.path.join(data_dir, filename)
    print(f"Loading {filename}...")
    return pd.read_csv(filepath, compression='gzip')

def sample_patients(n_patients=500):
    """Sample random patients and get their IDs."""
    patients_df = load_csv_gz('patients.csv.gz')
    
    total_patients = len(patients_df)
    print(f"\nTotal patients in dataset: {total_patients}")
    
    # Sample up to n_patients (or all if less than n_patients)
    n_sample = min(n_patients, total_patients)
    print(f"Sampling {n_sample} patients...")
    
    sampled_patients = patients_df.sample(n=n_sample, random_state=42)
    patient_ids = sampled_patients['subject_id'].tolist()
    
    print(f"Sampled {len(patient_ids)} patients")
    return patient_ids, sampled_patients

def filter_by_patients(df, patient_ids, patient_col='subject_id'):
    """Filter dataframe to only include sampled patients."""
    return df[df[patient_col].isin(patient_ids)]

def create_database():
    """Create database and schema."""
    print("\n=== Creating Database Schema ===")
    
    # Connect directly to the database (assume it already exists)
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Drop existing tables (in reverse order of dependencies)
    print("Dropping existing tables...")
    drop_tables = [
        'audit_logs', 'transfers', 'drgcodes', 'omr', 'prescriptions', 
        'diagnoses', 'admissions', 'patients',
        'ref_drg_codes', 'ref_opioid_drugs', 'ref_atc_codes', 'ref_icd_codes'
    ]
    for table in drop_tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    
    conn.commit()
    
    # Read and execute schema
    schema_path = os.path.join(os.path.dirname(__file__), 'schema_raw.sql')
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    cursor.execute(schema_sql)
    conn.commit()
    
    cursor.close()
    conn.close()
    
    print("✓ Database schema created")

def populate_reference_tables(conn):
    """Populate reference tables from demo data dictionary files."""
    print("\n=== Populating Reference Tables from Demo Dictionary Files ===")
    
    cursor = conn.cursor()
    
    # 1. Populate ref_icd_codes from d_icd_diagnoses.csv.gz (complete ICD dictionary)
    print("Populating ref_icd_codes from d_icd_diagnoses...")
    d_icd_diagnoses = load_csv_gz('d_icd_diagnoses.csv.gz', DEMO_DATA_DIR)
    
    records = []
    for _, row in d_icd_diagnoses.iterrows():
        icd_code = str(row['icd_code'])
        long_title = row.get('long_title', f"ICD Code {icd_code}")
        
        # Heuristic: pain if starts with M (musculoskeletal), G89 (pain), R52 (pain)
        # OUD if F11 (opioid use disorders)
        is_pain = (icd_code.startswith('M') or icd_code.startswith('G89') or 
                   icd_code.startswith('R52') or 'pain' in str(long_title).lower())
        is_oud = (icd_code.startswith('F11') or 'opioid' in str(long_title).lower())
        
        records.append((
            icd_code,
            long_title,
            'Diagnosis',
            is_pain,
            is_oud
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO ref_icd_codes (icd_code, description, category, is_pain_related, is_oud_related)
        VALUES %s
        ON CONFLICT (icd_code) DO NOTHING
        """,
        records
    )
    conn.commit()
    print(f"✓ Inserted {len(records)} ICD diagnosis codes")
    
    # 2. Also add ICD procedure codes from d_icd_procedures.csv.gz
    print("Populating ref_icd_codes from d_icd_procedures...")
    d_icd_procedures = load_csv_gz('d_icd_procedures.csv.gz', DEMO_DATA_DIR)
    
    records = []
    for _, row in d_icd_procedures.iterrows():
        icd_code = str(row['icd_code'])
        long_title = row.get('long_title', f"ICD Procedure {icd_code}")
        
        records.append((
            icd_code,
            long_title,
            'Procedure',
            False,  # procedures not pain-related
            False   # procedures not OUD-related
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO ref_icd_codes (icd_code, description, category, is_pain_related, is_oud_related)
        VALUES %s
        ON CONFLICT (icd_code) DO NOTHING
        """,
        records
    )
    conn.commit()
    print(f"✓ Inserted {len(records)} ICD procedure codes")
    
    # 3. Populate ref_drg_codes - collect all unique DRG codes from patient data
    print("Populating ref_drg_codes from patient DRG codes...")
    patient_drg = load_csv_gz('drgcodes.csv.gz', DATA_DIR)
    unique_drg = patient_drg[['drg_code', 'drg_type', 'drg_severity', 'drg_mortality']].drop_duplicates()
    
    records = []
    for _, row in unique_drg.iterrows():
        drg_severity = row.get('drg_severity', None)
        drg_mortality = row.get('drg_mortality', None)
        
        # Convert to int if not null, handle NaN
        if pd.notna(drg_severity):
            drg_severity = int(drg_severity)
        else:
            drg_severity = None
            
        if pd.notna(drg_mortality):
            drg_mortality = int(drg_mortality)
        else:
            drg_mortality = None
        
        records.append((
            str(row['drg_code']),
            f"DRG Code {row['drg_code']}",
            None,  # mdc_code
            drg_severity,
            drg_mortality
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO ref_drg_codes (drg_code, description, mdc_code, severity_level, mortality_risk)
        VALUES %s
        ON CONFLICT (drg_code) DO NOTHING
        """,
        records
    )
    conn.commit()
    print(f"✓ Inserted {len(records)} DRG codes")
    
    # 4. Add any missing ICD codes from patient data (fallback for codes not in dictionary)
    print("Adding missing ICD codes from patient data...")
    patient_diagnoses = load_csv_gz('diagnoses_icd.csv.gz', DATA_DIR)
    unique_patient_icd = patient_diagnoses[['icd_code', 'icd_version']].drop_duplicates()
    
    records = []
    for _, row in unique_patient_icd.iterrows():
        icd_code = str(row['icd_code'])
        icd_version = row['icd_version']
        
        # Simple heuristic for missing codes
        is_pain = (icd_code.startswith('M') or icd_code.startswith('G89') or icd_code.startswith('R52'))
        is_oud = icd_code.startswith('F11')
        
        records.append((
            icd_code,
            f"ICD-{icd_version} Code {icd_code}",
            'Diagnosis',
            is_pain,
            is_oud
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO ref_icd_codes (icd_code, description, category, is_pain_related, is_oud_related)
        VALUES %s
        ON CONFLICT (icd_code) DO NOTHING
        """,
        records
    )
    conn.commit()
    print(f"✓ Added {len(records)} codes from patient data (duplicates skipped)")
    
    cursor.close()
    print("✓ All reference tables populated")

def insert_patients(conn, patients_df):
    """Insert patients into patients table."""
    print("\nInserting patients...")
    
    cursor = conn.cursor()
    
    # Prepare records
    records = []
    for _, row in patients_df.iterrows():
        records.append((
            str(row['subject_id']),
            row['gender'],
            pd.to_datetime(row['anchor_year_group'].split(' - ')[0] + '-01-01'),  # Approximate DOB
            row.get('race', 'Unknown'),
            row.get('ethnicity', 'Unknown'),
            datetime.now()
        ))
    
    # Insert
    execute_values(
        cursor,
        """
        INSERT INTO patients (patient_id, gender, date_of_birth, race, ethnicity, updated_at)
        VALUES %s
        """,
        records
    )
    
    conn.commit()
    cursor.close()
    
    print(f"✓ Inserted {len(records)} patients")

def insert_admissions(conn, patient_ids):
    """Insert admissions for sampled patients."""
    print("\nInserting admissions...")
    
    admissions_df = load_csv_gz('admissions.csv.gz')
    admissions_df = filter_by_patients(admissions_df, patient_ids, 'subject_id')
    
    cursor = conn.cursor()
    
    records = []
    for _, row in admissions_df.iterrows():
        records.append((
            str(row['hadm_id']),
            str(row['subject_id']),
            pd.to_datetime(row['admittime']),
            pd.to_datetime(row['dischtime']) if pd.notna(row['dischtime']) else None,
            row.get('admission_type', 'UNKNOWN'),
            row.get('admission_location', 'UNKNOWN'),
            row.get('discharge_location', 'UNKNOWN'),
            row.get('insurance', 'UNKNOWN'),
            row.get('language', 'ENGLISH'),
            row.get('marital_status', 'UNKNOWN'),
            bool(int(row.get('hospital_expire_flag', 0)))
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO admissions 
        (admission_id, patient_id, admit_time, discharge_time, admission_type, 
         admission_location, discharge_location, insurance, language, marital_status, 
         hospital_expire_flag)
        VALUES %s
        """,
        records
    )
    
    conn.commit()
    cursor.close()
    
    print(f"✓ Inserted {len(records)} admissions")
    return admissions_df['hadm_id'].tolist()

def insert_diagnoses(conn, admission_ids):
    """Insert diagnoses for sampled admissions."""
    print("\nInserting diagnoses...")
    
    diagnoses_df = load_csv_gz('diagnoses_icd.csv.gz')
    diagnoses_df = filter_by_patients(diagnoses_df, admission_ids, 'hadm_id')
    
    cursor = conn.cursor()
    
    records = []
    for _, row in diagnoses_df.iterrows():
        records.append((
            str(row['subject_id']),
            str(row['hadm_id']),
            row['icd_code'],
            row.get('icd_version', 10),
            row.get('seq_num', 1)
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO diagnoses 
        (patient_id, admission_id, icd_code, icd_version, seq_num)
        VALUES %s
        """,
        records
    )
    
    conn.commit()
    cursor.close()
    
    print(f"✓ Inserted {len(records)} diagnoses")

def insert_prescriptions(conn, patient_ids):
    """Insert prescriptions for sampled patients."""
    print("\nInserting prescriptions (this may take a while)...")
    
    # Load prescriptions in chunks due to large file size
    prescriptions_df = load_csv_gz('prescriptions.csv.gz')
    prescriptions_df = filter_by_patients(prescriptions_df, patient_ids, 'subject_id')
    
    print(f"Found {len(prescriptions_df)} prescriptions for sampled patients")
    
    cursor = conn.cursor()
    
    records = []
    for _, row in prescriptions_df.iterrows():
        records.append((
            str(row['subject_id']),
            str(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
            row['drug'],
            row.get('drug_name_generic', None),  # generic_name
            row.get('ndc', None),  # ndc_code
            None,  # atc_code (not in data)
            pd.to_datetime(row['starttime']) if pd.notna(row.get('starttime')) else None,
            pd.to_datetime(row['stoptime']) if pd.notna(row.get('stoptime')) else None,
            row.get('dose_val_rx', None),
            row.get('dose_unit_rx', None),
            row.get('route', None),
            row.get('frequency', None)
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO prescriptions 
        (patient_id, admission_id, drug_name, generic_name, ndc_code, atc_code,
         start_time, stop_time, dose_val_rx, dose_unit_rx, route, frequency)
        VALUES %s
        ON CONFLICT DO NOTHING
        """,
        records,
        page_size=1000
    )
    
    conn.commit()
    cursor.close()
    
    print(f"✓ Inserted {len(records)} prescriptions")

def insert_omr(conn, patient_ids):
    """Insert OMR (vital signs, BMI) for sampled patients."""
    print("\nInserting OMR records...")
    
    omr_df = load_csv_gz('omr.csv.gz')
    omr_df = filter_by_patients(omr_df, patient_ids, 'subject_id')
    
    cursor = conn.cursor()
    
    records = []
    for _, row in omr_df.iterrows():
        records.append((
            str(row['subject_id']),
            pd.to_datetime(row['chartdate']) if pd.notna(row.get('chartdate')) else datetime.now(),
            row['result_name'],
            row.get('result_value', None)
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO omr 
        (patient_id, chart_time, result_name, result_value)
        VALUES %s
        """,
        records
    )
    
    conn.commit()
    cursor.close()
    
    print(f"✓ Inserted {len(records)} OMR records")

def insert_drgcodes(conn, admission_ids):
    """Insert DRG codes for sampled admissions."""
    print("\nInserting DRG codes...")
    
    drg_df = load_csv_gz('drgcodes.csv.gz')
    drg_df = filter_by_patients(drg_df, admission_ids, 'hadm_id')
    
    cursor = conn.cursor()
    
    records = []
    for _, row in drg_df.iterrows():
        records.append((
            str(row['subject_id']),
            str(row['hadm_id']),
            row['drg_code'],
            row.get('drg_type', 'MS'),
            row.get('drg_severity', None),
            row.get('drg_mortality', None)
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO drgcodes 
        (patient_id, admission_id, drg_code, drg_type, drg_severity, drg_mortality)
        VALUES %s
        """,
        records
    )
    
    conn.commit()
    cursor.close()
    
    print(f"✓ Inserted {len(records)} DRG codes")

def insert_transfers(conn, admission_ids):
    """Insert transfers for sampled admissions."""
    print("\nInserting transfers...")
    
    transfers_df = load_csv_gz('transfers.csv.gz')
    transfers_df = filter_by_patients(transfers_df, admission_ids, 'hadm_id')
    
    cursor = conn.cursor()
    
    records = []
    for _, row in transfers_df.iterrows():
        # Calculate LOS in hours
        in_time = pd.to_datetime(row['intime'])
        out_time = pd.to_datetime(row['outtime']) if pd.notna(row.get('outtime')) else None
        los_hours = None
        if out_time:
            los_hours = (out_time - in_time).total_seconds() / 3600
        
        # Determine careunit type
        care_unit = row.get('careunit', 'Unknown')
        if 'ICU' in care_unit or 'CCU' in care_unit:
            careunit_type = 'ICU'
        elif 'Emergency' in care_unit or 'ED' in care_unit:
            careunit_type = 'Emergency'
        else:
            careunit_type = 'Ward'
        
        records.append((
            str(row['subject_id']),
            str(row['hadm_id']),
            in_time,
            out_time,
            care_unit,
            careunit_type,
            los_hours
        ))
    
    execute_values(
        cursor,
        """
        INSERT INTO transfers 
        (patient_id, admission_id, in_time, out_time, care_unit, careunit_type, los_hours)
        VALUES %s
        """,
        records
    )
    
    conn.commit()
    cursor.close()
    
    print(f"✓ Inserted {len(records)} transfers")

def main():
    """Main execution."""
    print("=" * 70)
    print("POPULATING RAW EHR DATABASE")
    print("=" * 70)
    
    # Sample patients
    patient_ids, patients_df = sample_patients(n_patients=500)
    
    # Create database and schema
    create_database()
    
    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    
    try:
        # First populate reference tables from demo data
        populate_reference_tables(conn)
        
        # Then insert patient data from synthetic 50K data (respecting foreign keys)
        insert_patients(conn, patients_df)
        admission_ids = insert_admissions(conn, patient_ids)
        insert_diagnoses(conn, admission_ids)
        insert_prescriptions(conn, patient_ids)
        insert_omr(conn, patient_ids)
        insert_drgcodes(conn, admission_ids)
        insert_transfers(conn, admission_ids)
        
        print("\n" + "=" * 70)
        print("✓ DATABASE POPULATED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total patients: 500")
        print(f"Database: {DB_CONFIG['database']}")
        print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()
