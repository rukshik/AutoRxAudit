"""
Generate SQL UPDATE statements to populate first_name and last_name in autorxaudit patients table
"""

import re

# Read the patients_data.sql file with proper encoding
with open('../../doctor/database/patients_data.sql', 'r', encoding='utf-16') as f:
    content = f.read()

# Extract all INSERT statements using a pattern that handles line breaks
# Pattern: ('patient_id', 'first_name', 'last_name', 'date')
pattern = r"\('(\d+)',\s*'([^']+)',\s*'([^']+)',\s*'[\d-]+'\)"
matches = re.findall(pattern, content)

print(f"Found {len(matches)} matches")

# Generate UPDATE statements
with open('update_patient_names.sql', 'w') as f:
    f.write("-- Update patients table in autorxaudit database with first_name and last_name\n")
    f.write("-- Generated from doctor/database/patients_data.sql\n\n")
    f.write("-- Total: {} patients\n\n".format(len(matches)))
    
    for patient_id, first_name, last_name in matches:
        f.write(f"UPDATE patients SET first_name = '{first_name}', last_name = '{last_name}' WHERE patient_id = '{patient_id}';\n")
    
    f.write("\n-- All 500 patients updated\n")

print(f"Generated UPDATE statements for {len(matches)} patients")
print("File: update_patient_names.sql")
