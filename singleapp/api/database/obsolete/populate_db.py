"""
Generate 500 synthetic patients from our test dataset and populate PostgreSQL database.
Uses real feature distributions from our 50K v3 processed data.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'autorxaudit'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

# Feature columns based on metadata
ELIGIBILITY_FEATURES = [
    'avg_drg_severity', 'bmi', 'avg_drg_mortality', 'n_icu_admissions',
    'n_icu_stays', 'max_drg_severity', 'high_severity_flag', 'total_icu_hours',
    'obesity_flag', 'total_icu_days', 'atc_A_rx_count', 'n_admissions_with_drg',
    'n_hospital_admits', 'avg_los_days', 'total_los_days', 'has_bmi'
]

OUD_FEATURES = [
    'opioid_rx_count', 'distinct_opioids', 'opioid_exposure_days', 'opioid_hadms',
    'age_at_first_admit', 'total_icu_days', 'n_admissions_with_drg', 'n_hospital_admits',
    'avg_drg_severity', 'n_icu_admissions', 'any_opioid_flag', 'atc_A_rx_count',
    'total_icu_hours', 'avg_drg_mortality', 'total_los_days', 'n_icu_stays',
    'atc_C_rx_count', 'avg_los_days', 'atc_B_rx_count'
]

ALL_FEATURES = [
    'age_at_first_admit', 'bmi', 'has_bmi', 'obesity_flag',
    'n_hospital_admits', 'n_admissions_with_drg', 'n_icu_admissions', 'n_icu_stays',
    'total_los_days', 'avg_los_days', 'total_icu_days', 'total_icu_hours',
    'avg_drg_severity', 'max_drg_severity', 'avg_drg_mortality', 'high_severity_flag',
    'atc_A_rx_count', 'atc_B_rx_count', 'atc_C_rx_count',
    'opioid_rx_count', 'distinct_opioids', 'opioid_exposure_days', 'opioid_hadms', 'any_opioid_flag'
]


def load_test_data(data_dir='../../ai-layer/processed_data/50000_v3'):
    """Load test dataset with all features."""
    print(f"Loading test data from {data_dir}...")
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    print(f"Loaded {len(test_df)} test records")
    return test_df


def sample_patients(test_df, n_patients=500):
    """Sample diverse patients from test set."""
    print(f"\nSampling {n_patients} diverse patients...")
    
    # Sample randomly from test set
    # Features represent diverse patients with various severity levels
    sampled = test_df.sample(n=n_patients, random_state=42).reset_index(drop=True)
    
    print(f"  - Sampled {len(sampled)} patients")
    print(f"  - With opioid history: {sampled['any_opioid_flag'].sum()} patients")
    print(f"  - High severity: {sampled['high_severity_flag'].sum()} patients")
    
    return sampled


def prepare_patient_records(sampled_df):
    """Convert dataframe to patient records for insertion."""
    print("\nPreparing patient records...")
    
    records = []
    for idx, row in sampled_df.iterrows():
        patient_id = f"PAT_{idx+1:05d}"  # PAT_00001, PAT_00002, etc.
        
        record = {
            'patient_id': patient_id,
            'age_at_first_admit': float(row.get('age_at_first_admit', 0)),
            'bmi': float(row.get('bmi', 0)) if pd.notna(row.get('bmi')) else None,
            'has_bmi': bool(row.get('has_bmi', False)),
            'obesity_flag': bool(row.get('obesity_flag', False)),
            'n_hospital_admits': int(row.get('n_hospital_admits', 0)),
            'n_admissions_with_drg': int(row.get('n_admissions_with_drg', 0)),
            'n_icu_admissions': int(row.get('n_icu_admissions', 0)),
            'n_icu_stays': int(row.get('n_icu_stays', 0)),
            'total_los_days': float(row.get('total_los_days', 0)),
            'avg_los_days': float(row.get('avg_los_days', 0)),
            'total_icu_days': float(row.get('total_icu_days', 0)),
            'total_icu_hours': float(row.get('total_icu_hours', 0)),
            'avg_drg_severity': float(row.get('avg_drg_severity', 0)),
            'max_drg_severity': float(row.get('max_drg_severity', 0)),
            'avg_drg_mortality': float(row.get('avg_drg_mortality', 0)),
            'high_severity_flag': bool(row.get('high_severity_flag', False)),
            'atc_A_rx_count': int(row.get('atc_A_rx_count', 0)),
            'atc_B_rx_count': int(row.get('atc_B_rx_count', 0)),
            'atc_C_rx_count': int(row.get('atc_C_rx_count', 0)),
            'opioid_rx_count': int(row.get('opioid_rx_count', 0)),
            'distinct_opioids': int(row.get('distinct_opioids', 0)),
            'opioid_exposure_days': float(row.get('opioid_exposure_days', 0)),
            'opioid_hadms': int(row.get('opioid_hadms', 0)),
            'any_opioid_flag': bool(row.get('any_opioid_flag', False))
        }
        
        records.append(record)
    
    print(f"Prepared {len(records)} patient records")
    return records


def create_database(conn):
    """Create database schema."""
    print("\nCreating database schema...")
    
    with open('schema.sql', 'r') as f:
        schema_sql = f.read()
    
    cursor = conn.cursor()
    cursor.execute(schema_sql)
    conn.commit()
    cursor.close()
    print("Schema created successfully")


def insert_patients(conn, records):
    """Insert patient records into database."""
    print(f"\nInserting {len(records)} patients into database...")
    
    cursor = conn.cursor()
    
    # Clear existing data
    cursor.execute("TRUNCATE TABLE audit_logs CASCADE")
    cursor.execute("TRUNCATE TABLE patients CASCADE")
    
    # Prepare insert statement
    columns = list(records[0].keys())
    insert_sql = f"""
        INSERT INTO patients ({', '.join(columns)})
        VALUES ({', '.join(['%s'] * len(columns))})
    """
    
    # Convert records to tuples
    values = [tuple(r[col] for col in columns) for r in records]
    
    # Bulk insert
    cursor.executemany(insert_sql, values)
    conn.commit()
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM patients")
    count = cursor.fetchone()[0]
    cursor.close()
    
    print(f"Successfully inserted {count} patients")
    return count


def verify_data(conn):
    """Verify inserted data with statistics."""
    print("\n" + "="*60)
    print("DATABASE VERIFICATION")
    print("="*60)
    
    cursor = conn.cursor()
    
    # Total count
    cursor.execute("SELECT COUNT(*) FROM patients")
    total = cursor.fetchone()[0]
    print(f"\nTotal patients: {total}")
    
    # Sample records
    cursor.execute("""
        SELECT patient_id, age_at_first_admit, bmi, n_hospital_admits, 
               opioid_rx_count, avg_drg_severity
        FROM patients 
        LIMIT 5
    """)
    
    print("\nSample records:")
    print("-" * 100)
    print(f"{'Patient ID':<15} {'Age':<8} {'BMI':<8} {'Admits':<10} {'Opioids':<10} {'Severity':<10}")
    print("-" * 100)
    
    for row in cursor.fetchall():
        patient_id, age, bmi, admits, opioids, severity = row
        bmi_str = f"{bmi:.1f}" if bmi else "N/A"
        print(f"{patient_id:<15} {age:<8.1f} {bmi_str:<8} {admits:<10} {opioids:<10} {severity:<10.2f}")
    
    # Statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(age_at_first_admit) as avg_age,
            AVG(bmi) as avg_bmi,
            SUM(CASE WHEN opioid_rx_count > 0 THEN 1 ELSE 0 END) as with_opioids,
            SUM(CASE WHEN high_severity_flag THEN 1 ELSE 0 END) as high_severity,
            AVG(n_hospital_admits) as avg_admits
        FROM patients
    """)
    
    stats = cursor.fetchone()
    print("\nStatistics:")
    print("-" * 60)
    print(f"Average age: {stats[1]:.1f} years")
    print(f"Average BMI: {stats[2]:.1f}")
    print(f"Patients with opioid history: {stats[3]} ({stats[3]/stats[0]*100:.1f}%)")
    print(f"High severity cases: {stats[4]} ({stats[4]/stats[0]*100:.1f}%)")
    print(f"Average hospital admits: {stats[5]:.1f}")
    
    cursor.close()
    print("="*60)


def main():
    """Main execution."""
    print("="*60)
    print("AUTORXAUDIT - PATIENT DATABASE POPULATION")
    print("="*60)
    
    try:
        # Load test data
        test_df = load_test_data()
        
        # Sample 500 diverse patients
        sampled_df = sample_patients(test_df, n_patients=500)
        
        # Prepare records
        records = prepare_patient_records(sampled_df)
        
        # Connect to database
        print("\nConnecting to PostgreSQL...")
        print(f"  Host: {DB_CONFIG['host']}")
        print(f"  Database: {DB_CONFIG['database']}")
        print(f"  User: {DB_CONFIG['user']}")
        
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected successfully")
        
        # Create schema
        create_database(conn)
        
        # Insert patients
        insert_patients(conn, records)
        
        # Verify
        verify_data(conn)
        
        # Close connection
        conn.close()
        
        print("\n" + "="*60)
        print("POPULATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run the FastAPI application: uvicorn app:app --reload")
        print("  2. Test audit endpoint: POST /audit-prescription")
        print("  3. Query patients: GET /patients/{patient_id}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
