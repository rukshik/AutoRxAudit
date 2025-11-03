import psycopg2

# Base lists for generating unique names
first_names = ['Joe', 'Jane', 'Mary', 'Bob', 'Alice', 'Tom', 'Susan', 'David', 'Emily', 'Michael',
               'Sarah', 'John', 'Lisa', 'James', 'Linda', 'Robert', 'Patricia', 'William', 'Jennifer', 'Richard',
               'Charles', 'Barbara', 'Joseph', 'Elizabeth', 'Thomas', 'Margaret', 'Christopher', 'Dorothy', 'Daniel', 'Nancy',
               'Matthew', 'Karen', 'Anthony', 'Betty', 'Mark', 'Helen', 'Donald', 'Sandra', 'Steven', 'Donna',
               'Paul', 'Carol', 'Andrew', 'Ruth', 'Joshua', 'Sharon', 'Kenneth', 'Michelle', 'Kevin', 'Laura']

last_names = ['Doe', 'Smith', 'Johnson', 'Brown', 'Taylor', 'Jones', 'Wilson', 'Davis', 'Miller', 'Anderson',
              'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez',
              'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'King', 'Wright', 'Lopez', 'Hill',
              'Scott', 'Green', 'Adams', 'Baker', 'Nelson', 'Carter', 'Mitchell', 'Perez', 'Roberts', 'Turner',
              'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins', 'Stewart', 'Sanchez', 'Morris', 'Rogers']

def generate_unique_name(idx):
    """Generate unique first and last name for given index"""
    # For first 50, use base names
    if idx < len(first_names):
        first = first_names[idx]
    else:
        # After 50, add number suffix
        base_idx = idx % len(first_names)
        suffix = (idx // len(first_names)) + 1
        first = f"{first_names[base_idx]} {suffix}"
    
    # Similar for last names
    if idx < len(last_names):
        last = last_names[idx]
    else:
        base_idx = idx % len(last_names)
        suffix = (idx // len(last_names)) + 1
        last = f"{last_names[base_idx]} {suffix}"
    
    return first, last

# Connect to autorxaudit database
conn = psycopg2.connect(
    host='rxaudit.postgres.database.azure.com',
    port=5432,
    database='autorxaudit',
    user='posgres',
    password='UmaKiran12'
)

cur = conn.cursor()

# Get all patients with date_of_birth
cur.execute("""
    SELECT patient_id, date_of_birth
    FROM patients 
    ORDER BY patient_id
""")

patients = cur.fetchall()

print(f"-- {len(patients)} patients from autorxaudit database")
print("-- Auto-generated unique names with date_of_birth from existing data\n")
print("INSERT INTO patients (patient_id, first_name, last_name, date_of_birth) VALUES")

values = []
for idx, patient in enumerate(patients):
    patient_id = patient[0]
    date_of_birth = patient[1]
    
    first_name, last_name = generate_unique_name(idx)
    
    # Handle NULL date_of_birth
    if date_of_birth:
        dob_str = f"'{date_of_birth}'"
    else:
        dob_str = "NULL"
    
    values.append(f"('{patient_id}', '{first_name}', '{last_name}', {dob_str})")

print(",\n".join(values) + ";")

print(f"\n-- Total: {len(patients)} patients")

cur.close()
conn.close()
