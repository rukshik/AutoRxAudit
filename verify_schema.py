import psycopg2

def check_schema(db_name):
    print(f"\n{'='*70}")
    print(f"Schema Details: {db_name}")
    print('='*70)
    
    conn = psycopg2.connect(
        host='rxaudit.postgres.database.azure.com',
        port=5432,
        database=db_name,
        user='posgres',
        password='UmaKiran12'
    )
    
    cur = conn.cursor()
    
    # Get patients table columns
    print("\n📋 patients table:")
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'patients'
        ORDER BY ordinal_position
    """)
    for col in cur.fetchall():
        nullable = "NULL" if col[2] == 'YES' else "NOT NULL"
        print(f"  • {col[0]}: {col[1]} ({nullable})")
    
    # Get users table columns
    print("\n👤 users table:")
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'users'
        ORDER BY ordinal_position
    """)
    for col in cur.fetchall():
        nullable = "NULL" if col[2] == 'YES' else "NOT NULL"
        print(f"  • {col[0]}: {col[1]} ({nullable})")
    
    # Get prescription_requests table columns
    print("\n📝 prescription_requests table:")
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'prescription_requests'
        ORDER BY ordinal_position
    """)
    for col in cur.fetchall():
        nullable = "NULL" if col[2] == 'YES' else "NOT NULL"
        print(f"  • {col[0]}: {col[1]} ({nullable})")
    
    cur.close()
    conn.close()

print("\n" + "="*70)
print("DATABASE SCHEMA VERIFICATION")
print("="*70)

check_schema('doctor_office')
check_schema('pharmacy')

print("\n" + "="*70)
print("✅ Schema verification complete")
print("="*70)
