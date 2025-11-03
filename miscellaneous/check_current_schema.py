"""
Check current database schema and compare with schema files
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
print("Current Database Schema Analysis")
print("=" * 80)
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print()

conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Get all tables
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name
""")
tables = [row[0] for row in cursor.fetchall()]

print("CURRENT TABLES:")
print("-" * 80)
for table in tables:
    print(f"  - {table}")
print()

# Check prescriptions table structure
print("=" * 80)
print("PRESCRIPTIONS TABLE STRUCTURE")
print("=" * 80)
cursor.execute("""
    SELECT column_name, data_type, character_maximum_length, is_nullable
    FROM information_schema.columns 
    WHERE table_name='prescriptions' 
    ORDER BY ordinal_position
""")
columns = cursor.fetchall()
print("\nColumns:")
for col_name, data_type, max_len, nullable in columns:
    len_str = f"({max_len})" if max_len else ""
    null_str = "NULL" if nullable == 'YES' else "NOT NULL"
    print(f"  {col_name:25s} {data_type}{len_str:20s} {null_str}")

# Check row count
cursor.execute("SELECT COUNT(*) FROM prescriptions")
print(f"\nTotal rows: {cursor.fetchone()[0]}")

# Check sample data
cursor.execute("SELECT * FROM prescriptions LIMIT 1")
sample = cursor.fetchone()
cursor.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name='prescriptions' 
    ORDER BY ordinal_position
""")
col_names = [row[0] for row in cursor.fetchall()]
print("\nSample row:")
for i, col_name in enumerate(col_names):
    print(f"  {col_name}: {sample[i]}")

print()
print("=" * 80)
print("PRESCRIPTION_REQUESTS TABLE STRUCTURE")
print("=" * 80)
cursor.execute("""
    SELECT column_name, data_type, character_maximum_length, is_nullable
    FROM information_schema.columns 
    WHERE table_name='prescription_requests' 
    ORDER BY ordinal_position
""")
columns = cursor.fetchall()
if columns:
    print("\nColumns:")
    for col_name, data_type, max_len, nullable in columns:
        len_str = f"({max_len})" if max_len else ""
        null_str = "NULL" if nullable == 'YES' else "NOT NULL"
        print(f"  {col_name:25s} {data_type}{len_str:20s} {null_str}")
    
    cursor.execute("SELECT COUNT(*) FROM prescription_requests")
    print(f"\nTotal rows: {cursor.fetchone()[0]}")
else:
    print("Table does not exist!")

print()
print("=" * 80)
print("PATIENTS TABLE STRUCTURE")
print("=" * 80)
cursor.execute("""
    SELECT column_name, data_type, character_maximum_length, is_nullable
    FROM information_schema.columns 
    WHERE table_name='patients' 
    ORDER BY ordinal_position
""")
columns = cursor.fetchall()
if columns:
    print("\nColumns:")
    for col_name, data_type, max_len, nullable in columns:
        len_str = f"({max_len})" if max_len else ""
        null_str = "NULL" if nullable == 'YES' else "NOT NULL"
        print(f"  {col_name:25s} {data_type}{len_str:20s} {null_str}")
    
    cursor.execute("SELECT COUNT(*) FROM patients")
    print(f"\nTotal rows: {cursor.fetchone()[0]}")
else:
    print("Table does not exist!")

cursor.close()
conn.close()

print()
print("=" * 80)
print("DONE")
print("=" * 80)
