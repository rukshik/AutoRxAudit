import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from api/.env
api_dir = Path(__file__).parent.parent / 'api'
env_path = api_dir / '.env'
load_dotenv(env_path)

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'autorxaudit'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

print("=" * 80)
print("AutoRxAudit - Update Database Schema")
print("=" * 80)
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print()

try:
    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cursor = conn.cursor()
    print("✅ Connected!")
    print()
    
    # Drop old audit_actions table if exists
    print("Dropping old audit_actions table...")
    cursor.execute("DROP TABLE IF EXISTS audit_actions CASCADE;")
    print("✅ Old table dropped!")
    print()
    
    # Create prescription_requests table (for workflow, separate from MIMIC-IV prescriptions)
    print("Dropping old prescription_requests table...")
    cursor.execute("DROP TABLE IF EXISTS prescription_requests CASCADE;")
    print("✅ Old table dropped!")
    
    print("Creating prescription_requests table...")
    cursor.execute("""
        CREATE TABLE prescription_requests (
            prescription_id SERIAL PRIMARY KEY,
            patient_id VARCHAR(50) NOT NULL,
            drug_name VARCHAR(255) NOT NULL,
            prescriber_id INTEGER REFERENCES users(user_id),
            prescribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) DEFAULT 'PENDING'
        );
    """)
    print("✅ Prescription requests table created!")
    
    # Create indexes for prescription_requests
    print("Creating indexes for prescription_requests...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescription_requests_patient ON prescription_requests(patient_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescription_requests_status ON prescription_requests(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescription_requests_created ON prescription_requests(prescribed_at DESC);")
    print("✅ Prescription request indexes created!")
    print()
    
    # Drop old audit_logs table if exists (to recreate with new schema)
    print("Dropping old audit_logs table...")
    cursor.execute("DROP TABLE IF EXISTS audit_logs CASCADE;")
    print("✅ Old audit_logs dropped!")
    print()
    
    # Create new audit_logs table with pharmacist decision fields
    print("Creating new audit_logs table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            audit_id SERIAL PRIMARY KEY,
            prescription_id INTEGER REFERENCES prescription_requests(prescription_id),
            patient_id VARCHAR(50) NOT NULL,
            drug_name VARCHAR(255) NOT NULL,
            
            -- AI Model Results
            eligibility_score FLOAT NOT NULL,
            eligibility_prediction INTEGER NOT NULL,
            oud_risk_score FLOAT NOT NULL,
            oud_risk_prediction INTEGER NOT NULL,
            flagged BOOLEAN NOT NULL,
            flag_reason TEXT,
            recommendation TEXT,
            
            -- Pharmacist Decision (filled after review)
            reviewed_by INTEGER REFERENCES users(user_id),
            action VARCHAR(50),
            action_reason TEXT,
            reviewed_at TIMESTAMP,
            
            -- Timestamps
            audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("✅ Audit logs table created!")
    
    # Create indexes for audit_logs
    print("Creating indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_prescription ON audit_logs(prescription_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_patient ON audit_logs(patient_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_flagged ON audit_logs(flagged);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_reviewed_by ON audit_logs(reviewed_by);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_audited ON audit_logs(audited_at DESC);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_reviewed ON audit_logs(reviewed_at DESC);")
    print("✅ Indexes created!")
    print()
    
    cursor.close()
    conn.close()
    
    print("=" * 80)
    print("✅ Database schema updated successfully!")
    print("=" * 80)
    print()
    print("New workflow:")
    print("1. Submit prescription → Insert into prescription_requests table")
    print("2. AI audit → Insert into audit_logs (AI results only)")
    print("3. Pharmacist decision → Update audit_logs (add action, reviewed_by, reviewed_at)")
    print()
    print("Note: prescription_requests is separate from MIMIC-IV prescriptions table")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
