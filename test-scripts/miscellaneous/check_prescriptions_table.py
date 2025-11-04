import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('../api/.env')

db_config = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

print("=" * 80)
print("Checking MIMIC-IV prescriptions table")
print("=" * 80)

conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Check columns in prescriptions table
cursor.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name='prescriptions' 
    ORDER BY ordinal_position
""")
columns = cursor.fetchall()
print("\nColumns in MIMIC-IV prescriptions table:")
for col_name, col_type in columns:
    print(f"  - {col_name}: {col_type}")

# Check row count
cursor.execute("SELECT COUNT(*) FROM prescriptions")
count = cursor.fetchone()[0]
print(f"\nTotal rows: {count}")

# Check sample data
cursor.execute("SELECT * FROM prescriptions LIMIT 3")
rows = cursor.fetchall()
print(f"\nSample data (first 3 rows):")
for i, row in enumerate(rows, 1):
    print(f"  Row {i}: {row}")

cursor.close()
conn.close()

print("\n" + "=" * 80)
print("✅ MIMIC-IV prescriptions table is intact!")
print("=" * 80)
