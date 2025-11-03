import psycopg2

# Database connection
conn = psycopg2.connect(
    host="rxaudit.postgres.database.azure.com",
    port=5432,
    database="autorxaudit",
    user="posgres",
    password="UmaKiran12"
)

cursor = conn.cursor()

# Check how many patients have names populated
cursor.execute("""
    SELECT 
        COUNT(*) as total_patients,
        COUNT(first_name) as with_first_name,
        COUNT(last_name) as with_last_name
    FROM patients
""")

result = cursor.fetchone()
print(f"Total patients: {result[0]}")
print(f"Patients with first_name: {result[1]}")
print(f"Patients with last_name: {result[2]}")

# Show a few sample patients
cursor.execute("""
    SELECT patient_id, first_name, last_name
    FROM patients
    WHERE first_name IS NOT NULL
    LIMIT 10
""")

print("\nSample patients:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} {row[2]}")

cursor.close()
conn.close()
