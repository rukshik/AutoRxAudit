import psycopg2
from dotenv import load_dotenv
import os
import requests

load_dotenv('api/.env')

db_config = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}
API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000')

conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# Find top patients by opioid prescription count
cur.execute("""
    SELECT patient_id, COUNT(*) as opioid_count
    FROM prescriptions
    WHERE drug ~* 'oxycodone|hydrocodone|morphine|fentanyl|codeine|tramadol|hydromorphone|oxymorphone|methadone|buprenorphine|meperidine|tapentadol'
    GROUP BY patient_id
    ORDER BY opioid_count DESC
    LIMIT 10
""")
rows = cur.fetchall()

if not rows:
    print('No patients with opioid prescriptions found')
    exit(0)

print('Top patients by opioid prescriptions:')
patient_ids = [r[0] for r in rows]
for pid, cnt in rows:
    print(f"  {pid}: {cnt}")

print('\nCalling API /audit-prescription for each patient with Oxycodone 5mg...')
results = []
for pid in patient_ids:
    payload = {
        'patient_id': pid,
        'drug_name': 'Oxycodone 5mg',
        'prescriber_id': 'TEST',
        'quantity': 30,
        'days_supply': 5
    }
    try:
        r = requests.post(f'{API_URL}/audit-prescription', json=payload, timeout=15)
        if r.status_code == 200:
            data = r.json()
            results.append((pid, data.get('eligibility_score'), data.get('eligibility_prediction'), data.get('oud_risk_score'), data.get('oud_risk_prediction'), data.get('flagged')))
        else:
            results.append((pid, 'API_ERROR', r.status_code))
    except Exception as e:
        results.append((pid, 'REQUEST_ERROR', str(e)))

print('\nResults:')
for res in results:
    print(res)

cur.close()
conn.close()
