"""
Comprehensive fix for prescriptions table to match schema_raw.sql expectations
Adds compatibility columns to map MIMIC-IV names to application names
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
print("Comprehensive Fix: Prescriptions Table Column Mapping")
print("=" * 80)
print(f"Host: {db_config['host']}")
print(f"Database: {db_config['database']}")
print()

conn = psycopg2.connect(**db_config)
conn.autocommit = True
cursor = conn.cursor()

# Get current columns
cursor.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name='prescriptions'
    ORDER BY ordinal_position
""")
current_columns = [row[0] for row in cursor.fetchall()]
print("Current columns:", ', '.join(current_columns))
print()

columns_to_add = []

# Check and add missing columns
print("Checking required columns...")
print("-" * 80)

# 1. admission_id
if 'admission_id' not in current_columns:
    print("❌ admission_id missing")
    columns_to_add.append(('admission_id', 'hadm_id'))
else:
    print("✅ admission_id exists")

# 2. start_time
if 'start_time' not in current_columns:
    print("❌ start_time missing")
    columns_to_add.append(('start_time', 'starttime'))
else:
    print("✅ start_time exists")

# 3. stop_time
if 'stop_time' not in current_columns:
    print("❌ stop_time missing")
    columns_to_add.append(('stop_time', 'stoptime'))
else:
    print("✅ stop_time exists")

print()

if columns_to_add:
    print("Adding missing columns...")
    print("-" * 80)
    
    for new_col, source_col in columns_to_add:
        # Determine data type based on source column
        cursor.execute(f"""
            SELECT data_type, character_maximum_length
            FROM information_schema.columns 
            WHERE table_name='prescriptions' AND column_name='{source_col}'
        """)
        result = cursor.fetchone()
        if result:
            data_type = result[0]
            max_len = result[1]
            
            if data_type == 'timestamp without time zone':
                col_def = 'TIMESTAMP'
            elif data_type == 'character varying':
                col_def = f'VARCHAR({max_len})'
            else:
                col_def = 'TEXT'
            
            print(f"Adding {new_col} ({col_def}) from {source_col}...")
            cursor.execute(f"""
                ALTER TABLE prescriptions 
                ADD COLUMN {new_col} {col_def}
            """)
            print(f"  ✅ Column added")
            
            # Populate from source column
            print(f"  Populating {new_col} from {source_col}...")
            cursor.execute(f"""
                UPDATE prescriptions 
                SET {new_col} = {source_col}
            """)
            rows_updated = cursor.rowcount
            print(f"  ✅ Updated {rows_updated} rows")
            
            # Create index
            if new_col in ['admission_id', 'start_time', 'stop_time']:
                print(f"  Creating index on {new_col}...")
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_prescriptions_{new_col.replace('_', '')} 
                    ON prescriptions({new_col})
                """)
                print(f"  ✅ Index created")
            
            print()
else:
    print("✅ All required columns exist!")
    print()

# Verify final schema
print("=" * 80)
print("Final Verification")
print("=" * 80)

cursor.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(DISTINCT patient_id) as unique_patients,
        COUNT(admission_id) as with_admission,
        COUNT(start_time) as with_start_time
    FROM prescriptions
""")
total, unique_pat, with_adm, with_start = cursor.fetchone()
print(f"Total rows: {total}")
print(f"Unique patients: {unique_pat}")
print(f"Rows with admission_id: {with_adm}")
print(f"Rows with start_time: {with_start}")

# Show column mapping
print()
print("Column Mapping (MIMIC-IV → Application):")
print("-" * 80)
print("  subject_id → patient_id")
print("  hadm_id → admission_id")
print("  drug → drug_name (both exist)")
print("  starttime → start_time")
print("  stoptime → stop_time")
print("  generic_name (already exists)")

# Sample data
print()
cursor.execute("""
    SELECT prescription_id, patient_id, admission_id, drug_name, start_time
    FROM prescriptions 
    LIMIT 3
""")
print("Sample data:")
for row in cursor.fetchall():
    print(f"  ID:{row[0]} | patient:{row[1]} | admission:{row[2]} | drug:{row[3]} | start:{row[4]}")

cursor.close()
conn.close()

print()
print("=" * 80)
print("✅ Prescriptions table fully compatible with schema_raw.sql!")
print("=" * 80)
