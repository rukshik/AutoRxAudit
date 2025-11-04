import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'autorxaudit'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

print("=" * 80)
print("AutoRxAudit - Database Schema Setup")
print("=" * 80)
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print(f"User: {db_config['user']}")
print()

try:
    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cursor = conn.cursor()
    print("✅ Connected successfully!")
    print()
    
    # Create users table
    print("Creating users table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            full_name VARCHAR(255),
            role VARCHAR(50) DEFAULT 'clinician',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        );
    """)
    print("✅ Users table created!")
    
    # Create index on email
    print("Creating index on users.email...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """)
    print("✅ Index created!")
    print()
    
    # Create audit_actions table (without foreign key to audit_logs for now)
    print("Creating audit_actions table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_actions (
            action_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            patient_id VARCHAR(50) NOT NULL,
            drug_name VARCHAR(255) NOT NULL,
            
            -- AI Model Results
            flagged BOOLEAN NOT NULL,
            eligibility_score FLOAT,
            eligibility_prediction INTEGER,
            oud_risk_score FLOAT,
            oud_risk_prediction INTEGER,
            flag_reason TEXT,
            
            -- User Action
            action VARCHAR(50) NOT NULL,
            action_reason TEXT,
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("✅ Audit actions table created!")
    
    # Create indexes on audit_actions
    print("Creating indexes on audit_actions...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_actions_user ON audit_actions(user_id);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_actions_patient ON audit_actions(patient_id);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_actions_created ON audit_actions(created_at DESC);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_actions_action ON audit_actions(action);
    """)
    print("✅ Indexes created!")
    print()
    
    # Insert sample users
    print("Inserting demo users...")
    cursor.execute("""
        INSERT INTO users (email, password, full_name, role) VALUES 
            ('doctor@hospital.com', 'password123', 'Dr. Sarah Johnson', 'doctor'),
            ('pharmacist@hospital.com', 'password123', 'John Smith RPh', 'pharmacist'),
            ('admin@hospital.com', 'admin123', 'Admin User', 'admin')
        ON CONFLICT (email) DO NOTHING;
    """)
    print("✅ Demo users inserted!")
    print()
    
    # Verify users
    cursor.execute("SELECT user_id, email, full_name, role FROM users ORDER BY user_id;")
    users = cursor.fetchall()
    
    print("=" * 80)
    print("Demo Users Created:")
    print("=" * 80)
    print(f"{'ID':<5} {'Email':<30} {'Name':<25} {'Role':<15}")
    print("-" * 80)
    for user in users:
        print(f"{user[0]:<5} {user[1]:<30} {user[2]:<25} {user[3]:<15}")
    print()
    print("Demo Credentials:")
    print("  - doctor@hospital.com / password123")
    print("  - pharmacist@hospital.com / password123")
    print("  - admin@hospital.com / admin123")
    print()
    
    cursor.close()
    conn.close()
    
    print("=" * 80)
    print("✅ Database setup completed successfully!")
    print("=" * 80)
    print()
    print("You can now:")
    print("1. Start the backend: python -m uvicorn app:app --reload")
    print("2. Start the frontend: cd frontend && npm start")
    print("3. Login with: doctor@hospital.com / password123")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Troubleshooting:")
    print("1. Check .env file has correct database credentials")
    print("2. Ensure database server is accessible")
    print("3. Install psycopg2: pip install psycopg2-binary")
