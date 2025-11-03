import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv('api/.env')

# Check DB
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

patient_id = '20000199'

print("=" * 80)
print(f"Analysis for Patient {patient_id}")
print("=" * 80)

# Check if patient exists
cur.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
patient = cur.fetchone()
print(f"\nPatient in DB: {'YES' if patient else 'NO'}")
if patient:
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='patients' ORDER BY ordinal_position")
    cols = [r[0] for r in cur.fetchall()]
    print(f"Patient data: {dict(zip(cols, patient))}")

# Check opioid prescriptions
cur.execute("""
    SELECT drug_name, start_time 
    FROM prescriptions 
    WHERE patient_id = %s 
    AND drug_name ~* 'oxycodone|hydrocodone|morphine|fentanyl|codeine|tramadol|hydromorphone|oxymorphone|methadone|buprenorphine|meperidine|tapentadol'
    ORDER BY start_time
""", (patient_id,))
opioid_rxs = cur.fetchall()
print(f"\nOpioid prescriptions: {len(opioid_rxs)}")
for drug, date in opioid_rxs[:5]:
    print(f"  - {drug}: {date}")

# Check all prescriptions
cur.execute("SELECT COUNT(*) FROM prescriptions WHERE patient_id = %s", (patient_id,))
total_rx = cur.fetchone()[0]
print(f"Total prescriptions: {total_rx}")

# Check admissions
cur.execute("SELECT COUNT(*) FROM admissions WHERE patient_id = %s", (patient_id,))
admissions = cur.fetchone()[0]
print(f"Admissions: {admissions}")

# Check diagnoses  
cur.execute("SELECT COUNT(*) FROM diagnoses WHERE patient_id = %s", (patient_id,))
diagnoses = cur.fetchone()[0]
print(f"Diagnoses: {diagnoses}")

cur.close()
conn.close()

# Check training data
print("\n" + "=" * 80)
print("Training Data Check")
print("=" * 80)
try:
    df = pd.read_csv('processed_data/50000_v3/full_data_with_labels.csv')
    if patient_id in df['patient_id'].values:
        patient_row = df[df['patient_id'] == patient_id].iloc[0]
        print(f"Patient {patient_id} in training data:")
        print(f"  y_oud: {patient_row['y_oud']}")
        print(f"  y_eligibility: {patient_row.get('y_eligibility', 'N/A')}")
        
        # Show key OUD features
        oud_features = ['opioid_rx_count', 'distinct_opioids', 'opioid_exposure_days', 'opioid_hadms']
        print("\n  Key OUD features:")
        for feat in oud_features:
            if feat in patient_row:
                print(f"    {feat}: {patient_row[feat]}")
    else:
        print(f"Patient {patient_id} NOT in training data!")
except Exception as e:
    print(f"Error reading training data: {e}")
