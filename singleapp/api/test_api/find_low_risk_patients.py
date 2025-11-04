"""
Find patients with low OUD risk by analyzing their features.
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# Find patients with:
# 1. Few or no opioid prescriptions in history
# 2. Low severity scores
# 3. Minimal ICU stays
# 4. Few admissions

print("Finding patients with LOW OUD risk indicators...")
print("=" * 80)
print()

cursor.execute("""
    WITH patient_opioid_count AS (
        SELECT 
            patient_id,
            COUNT(*) as opioid_rx_count
        FROM prescriptions
        WHERE LOWER(drug_name) IN (SELECT LOWER(drug_name) FROM ref_opioid_drugs)
        GROUP BY patient_id
    ),
    patient_stats AS (
        SELECT 
            p.patient_id,
            COUNT(DISTINCT a.admission_id) as n_admissions,
            COALESCE(poc.opioid_rx_count, 0) as opioid_rx_count,
            COALESCE(AVG(d.drg_severity::float), 0) as avg_severity
        FROM patients p
        LEFT JOIN admissions a ON p.patient_id = a.patient_id
        LEFT JOIN drgcodes d ON a.admission_id = d.admission_id
        LEFT JOIN patient_opioid_count poc ON p.patient_id = poc.patient_id
        GROUP BY p.patient_id, poc.opioid_rx_count
    )
    SELECT 
        patient_id,
        n_admissions,
        opioid_rx_count,
        ROUND(avg_severity::numeric, 2) as avg_severity
    FROM patient_stats
    WHERE opioid_rx_count = 0  -- No prior opioid prescriptions
        AND n_admissions <= 3   -- Few admissions
    ORDER BY n_admissions, avg_severity
    LIMIT 20
""")

low_risk_patients = cursor.fetchall()

if low_risk_patients:
    print(f"Found {len(low_risk_patients)} LOW RISK patients:")
    print()
    print(f"{'Patient ID':<15} {'Admissions':<12} {'Opioid Rx':<12} {'Avg Severity':<14}")
    print("-" * 70)
    
    for row in low_risk_patients:
        patient_id, n_admissions, opioid_rx, avg_sev = row
        print(f"{patient_id:<15} {n_admissions:<12} {opioid_rx:<12} {avg_sev:<14}")
    
    print()
    print("=" * 80)
    print(f"Recommendation: Test with patient IDs like {low_risk_patients[0][0]}, {low_risk_patients[1][0]}, etc.")
else:
    print("No low-risk patients found with these criteria.")
    print("Trying less strict criteria...")
    
    cursor.execute("""
        WITH patient_stats AS (
            SELECT 
                p.patient_id,
                COUNT(DISTINCT a.hadm_id) as n_admissions,
                COALESCE(COUNT(DISTINCT pr.prescription_id), 0) as total_rx
            FROM patients p
            LEFT JOIN admissions a ON p.patient_id = a.patient_id
            LEFT JOIN prescriptions pr ON p.patient_id = pr.patient_id
            GROUP BY p.patient_id
        )
        SELECT 
            patient_id,
            n_admissions,
            total_rx
        FROM patient_stats
        WHERE n_admissions <= 2
        ORDER BY n_admissions, total_rx
        LIMIT 10
    """)
    
    minimal_patients = cursor.fetchall()
    print(f"\nPatients with minimal healthcare utilization:")
    for row in minimal_patients:
        print(f"  {row[0]} - {row[1]} admissions, {row[2]} prescriptions")

conn.close()
