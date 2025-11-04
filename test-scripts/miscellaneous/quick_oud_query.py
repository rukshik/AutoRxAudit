import psycopg2
from dotenv import load_dotenv
import os

load_dotenv('api/.env')
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()
# First check what columns exist
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='patients'")
cols = [r[0] for r in cur.fetchall()]
print(f"Patients table columns: {cols}")
print()

# Find patients with many opioid prescriptions (high OUD risk)
cur.execute("""
    SELECT p.patient_id, COUNT(*) as opioid_count
    FROM prescriptions p
    WHERE p.drug_name ~* 'oxycodone|hydrocodone|morphine|fentanyl|codeine'
    GROUP BY p.patient_id
    HAVING COUNT(*) > 3
    ORDER BY COUNT(*) DESC
    LIMIT 5
""")
results = cur.fetchall()
print('Patients with multiple opioid prescriptions (HIGH OUD RISK):')
for row in results:
    print(f"  {row[0]} - {row[1]} opioid prescriptions")
conn.close()
