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

# Read and execute the UPDATE statements one by one
with open('update_patient_names.sql', 'r', encoding='utf-8') as f:
    statements = [line.strip() for line in f if line.strip() and not line.strip().startswith('--')]

total_updated = 0
for statement in statements:
    cursor.execute(statement)
    total_updated += cursor.rowcount

conn.commit()

print(f"Successfully updated {total_updated} patients")

cursor.close()
conn.close()
