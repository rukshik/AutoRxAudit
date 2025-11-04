import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

# Load training data
df = pd.read_csv('ai-layer/processed_data/50000_v3/full_data_selected_features.csv')

print(f"Columns in file: {list(df.columns)}")
print(f"Shape: {df.shape}")
print()

# Check for patient_id or subject_id column
if 'patient_id' in df.columns:
    id_col = 'patient_id'
elif 'subject_id' in df.columns:
    id_col = 'subject_id'
else:
    id_col = df.columns[0]  # Use first column

print(f"Using ID column: {id_col}")

# Find patients with y_oud=1
high_oud = df[df['y_oud'] == 1][id_col].head(10).tolist()

print('Patients with y_oud=1 (HIGH OUD RISK) from training data:')
for pid in high_oud:
    print(f'  {pid}')

# Check if these patients exist in database
load_dotenv('api/.env')
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

print('\nChecking which patients exist in database...')

# Get all patient IDs from database
cur.execute("SELECT patient_id FROM patients LIMIT 500")
db_patients = set([str(r[0]) for r in cur.fetchall()])
print(f"Found {len(db_patients)} patients in database")

# Check overlap with training data
df_patients = set(df[id_col].astype(str).tolist())
overlap = db_patients.intersection(df_patients)
print(f"Overlap with training data: {len(overlap)} patients")

# Find high OUD risk patients that exist in DB
high_oud_in_db = []
for pid in high_oud:
    if str(pid) in db_patients:
        high_oud_in_db.append(pid)
        print(f'  ✅ {pid} exists in DB and has y_oud=1')

if high_oud_in_db:
    print(f'\n🎯 Recommended HIGH OUD RISK patient: {high_oud_in_db[0]}')
else:
    print('\n⚠️ No high OUD patients from training data exist in current DB')
    print('Checking for any patients with high opioid exposure...')
    
    # Check patients in DB that have high OUD in training data
    for pid in df[df['y_oud'] == 1][id_col].head(100):
        if str(pid) in db_patients:
            print(f'  Found: {pid}')
            high_oud_in_db.append(pid)
            if len(high_oud_in_db) >= 5:
                break
    
    if high_oud_in_db:
        print(f'\n🎯 Recommended HIGH OUD RISK patient: {high_oud_in_db[0]}')

conn.close()
