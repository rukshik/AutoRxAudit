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

print("Connecting to database:")
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print(f"User: {db_config['user']}")
print()

try:
    # Connect to database
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'users'
        );
    """)
    table_exists = cursor.fetchone()[0]
    
    if table_exists:
        print("✅ Users table exists!")
        print()
        
        # Query users
        cursor.execute("""
            SELECT user_id, email, full_name, role, is_active, 
                   TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') as created_at
            FROM users
            ORDER BY user_id;
        """)
        
        users = cursor.fetchall()
        
        if users:
            print(f"Found {len(users)} user(s):")
            print("-" * 100)
            print(f"{'ID':<5} {'Email':<30} {'Name':<25} {'Role':<15} {'Active':<8} {'Created':<20}")
            print("-" * 100)
            
            for user in users:
                user_id, email, full_name, role, is_active, created_at = user
                active_str = "Yes" if is_active else "No"
                print(f"{user_id:<5} {email:<30} {full_name:<25} {role:<15} {active_str:<8} {created_at:<20}")
        else:
            print("⚠️  Users table is empty - no users found!")
            print()
            print("Run this to create demo users:")
            print("psql -h rxaudit.postgres.database.azure.com -U posgres -d autorxaudit -f database/schema_with_users.sql")
    else:
        print("❌ Users table does NOT exist!")
        print()
        print("You need to create it by running:")
        print("psql -h rxaudit.postgres.database.azure.com -U posgres -d autorxaudit -f database/schema_with_users.sql")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Make sure:")
    print("1. Database credentials in .env are correct")
    print("2. Database server is accessible")
    print("3. psycopg2 is installed: pip install psycopg2-binary")
