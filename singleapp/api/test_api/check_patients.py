import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()
cur.execute('SELECT patient_id FROM patients ORDER BY patient_id LIMIT 20')
print('Available patients:')
for row in cur.fetchall():
    print(f"  {row[0]}")

conn.close()
