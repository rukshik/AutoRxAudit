"""Check database schema for prescriptions table"""
import psycopg2

DB_CONFIG = {
    'host': 'autorxaudit-server.postgres.database.azure.com',
    'port': 5432,
    'database': 'mimiciv_demo_raw',
    'user': 'cloudsa',
    'password': 'Poornima@1985'
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("Prescriptions table columns:")
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='prescriptions'")
for row in cur.fetchall():
    print(f"  {row[0]}")

print("\nSample prescription record:")
cur.execute("SELECT * FROM prescriptions LIMIT 1")
cols = [desc[0] for desc in cur.description]
row = cur.fetchone()
if row:
    for col, val in zip(cols, row):
        print(f"  {col}: {val}")

print("\nChecking for atc_code column:")
cur.execute("SELECT COUNT(*), COUNT(atc_code) FROM prescriptions")
total, with_atc = cur.fetchone()
print(f"  Total prescriptions: {total}")
print(f"  With atc_code: {with_atc}")

cur.close()
conn.close()
