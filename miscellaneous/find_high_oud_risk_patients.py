"""
Find patients with high OUD risk
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
print("Finding Patients with High OUD Risk Indicators")
print("=" * 80)

conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Find patients with multiple opioid prescriptions (high OUD risk indicator)
cursor.execute("""
    SELECT 
        patient_id,
        COUNT(*) as opioid_rx_count,
        COUNT(DISTINCT drug_name) as distinct_opioids,
        MIN(start_time) as first_rx,
        MAX(start_time) as last_rx
    FROM prescriptions
    WHERE drug_name ~* 'oxycodone|hydrocodone|morphine|fentanyl|codeine|tramadol|hydromorphone|oxymorphone|methadone|buprenorphine|meperidine|tapentadol'
    GROUP BY patient_id
    HAVING COUNT(*) > 5
    ORDER BY COUNT(*) DESC
    LIMIT 10
""")

print("\nPatients with multiple opioid prescriptions:")
print("-" * 80)
results = cursor.fetchall()
for row in results:
    patient_id, rx_count, distinct_opioids, first_rx, last_rx = row
    print(f"Patient {patient_id}:")
    print(f"  - {rx_count} opioid prescriptions")
    print(f"  - {distinct_opioids} different opioids")
    print(f"  - First Rx: {first_rx}")
    print(f"  - Last Rx: {last_rx}")
    print()

if results:
    high_risk_patient = results[0][0]
    print("=" * 80)
    print(f"Recommended HIGH OUD RISK patient: {high_risk_patient}")
    print("=" * 80)
    
    # Show their opioid prescription details
    cursor.execute("""
        SELECT drug_name, start_time, stop_time
        FROM prescriptions
        WHERE patient_id = %s
        AND drug_name ~* 'oxycodone|hydrocodone|morphine|fentanyl|codeine|tramadol|hydromorphone|oxymorphone|methadone|buprenorphine|meperidine|tapentadol'
        ORDER BY start_time
    """, (high_risk_patient,))
    
    print(f"\nOpioid prescriptions for {high_risk_patient}:")
    for drug, start, stop in cursor.fetchall():
        print(f"  - {drug}: {start} to {stop}")

cursor.close()
conn.close()
