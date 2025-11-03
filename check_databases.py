import psycopg2

def check_database(db_name):
    print(f"\n{'='*60}")
    print(f"Checking {db_name} database")
    print('='*60)
    
    try:
        conn = psycopg2.connect(
            host='rxaudit.postgres.database.azure.com',
            port=5432,
            database=db_name,
            user='posgres',
            password='UmaKiran12'
        )
        
        cur = conn.cursor()
        
        # Check tables
        print("\n📋 Tables:")
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        for table in tables:
            print(f"  ✓ {table[0]}")
        
        # Check patients count
        print("\n👥 Patients:")
        cur.execute("SELECT COUNT(*) FROM patients")
        patient_count = cur.fetchone()[0]
        print(f"  Total: {patient_count} patients")
        
        # Sample 5 patients
        cur.execute("SELECT patient_id, first_name, last_name, date_of_birth FROM patients LIMIT 5")
        sample_patients = cur.fetchall()
        print("  Sample records:")
        for p in sample_patients:
            print(f"    {p[0]}: {p[1]} {p[2]} (DOB: {p[3]})")
        
        # Check for duplicates
        cur.execute("""
            SELECT first_name, last_name, COUNT(*) as count
            FROM patients
            GROUP BY first_name, last_name
            HAVING COUNT(*) > 1
        """)
        duplicates = cur.fetchall()
        if duplicates:
            print(f"  ⚠️  Found {len(duplicates)} duplicate names!")
            for dup in duplicates[:5]:
                print(f"    {dup[0]} {dup[1]}: {dup[2]} times")
        else:
            print("  ✓ No duplicate names found")
        
        # Check users count
        print("\n👤 Users:")
        cur.execute("SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]
        print(f"  Total: {user_count} users")
        
        cur.execute("SELECT full_name, email, role FROM users ORDER BY role")
        users = cur.fetchall()
        print("  User list:")
        for u in users:
            print(f"    {u[0]} ({u[2]}): {u[1]}")
        
        # Check prescription_requests
        print("\n📝 Prescription Requests:")
        cur.execute("SELECT COUNT(*) FROM prescription_requests")
        rx_count = cur.fetchone()[0]
        print(f"  Total: {rx_count} prescriptions")
        
        if rx_count > 0:
            cur.execute("SELECT prescription_uuid, patient_id, drug_name, status FROM prescription_requests LIMIT 3")
            rxs = cur.fetchall()
            print("  Sample prescriptions:")
            for rx in rxs:
                print(f"    {rx[0]}: Patient {rx[1]} - {rx[2]} (Status: {rx[3]})")
        
        cur.close()
        conn.close()
        
        print(f"\n✅ {db_name} database is operational")
        
    except psycopg2.Error as e:
        print(f"\n❌ Error connecting to {db_name}:")
        print(f"  {e}")
        return False
    
    return True

# Check both databases
print("\n" + "="*60)
print("DATABASE HEALTH CHECK")
print("="*60)

doctor_ok = check_database('doctor_office')
pharmacy_ok = check_database('pharmacy')

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"doctor_office: {'✅ OK' if doctor_ok else '❌ ERROR'}")
print(f"pharmacy: {'✅ OK' if pharmacy_ok else '❌ ERROR'}")
print("="*60)
