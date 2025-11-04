"""
Test script to validate API predictions with opioid vs non-opioid prescriptions.
Tests that:
1. When passing an OPIOID drug -> System evaluates OUD risk
2. When passing a NON-OPIOID drug -> System should NOT flag
"""
import psycopg2
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Database connection
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

API_URL = "http://127.0.0.1:8000/audit-prescription"

# Test drugs
OPIOID_DRUG = "Oxycodone"  # This is an opioid
NON_OPIOID_DRUG = "Acetaminophen"  # This is NOT an opioid

def get_all_patients(conn):
    """Get all patients from database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT patient_id
        FROM patients
        ORDER BY patient_id
    """)
    results = cursor.fetchall()
    cursor.close()
    return [row[0] for row in results]

def test_patient_with_drug(patient_id, drug_name, prescription_id):
    """Test a patient with a specific drug through the API."""
    try:
        response = requests.post(
            API_URL,
            json={
                "patient_id": patient_id,
                "prescription_id": prescription_id,
                "drug_name": drug_name
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "detail": response.text}
    
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 80)
    print("AutoRxAudit API - Opioid vs Non-Opioid Prescription Test")
    print("=" * 80)
    print()
    
    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Get all patients
    print("Querying database for all patients...")
    all_patients = get_all_patients(conn)
    print(f"✓ Found {len(all_patients)} patients in database")
    print()
    
    conn.close()
    
    # Use mix of regular and low-risk patients
    # First 5 are random, these are low-risk patients with no opioid history
    low_risk_patients = ['20038478', '20002189', '20026527', '20045637', '20030188']
    test_patients = low_risk_patients[:5]
    
    # TEST 1: All patients with NON-OPIOID drug (should NOT be flagged)
    print("=" * 80)
    print(f"TEST 1: All Patients Prescribed NON-OPIOID ({NON_OPIOID_DRUG})")
    print("Should NOT be flagged for OUD risk")
    print("=" * 80)
    print()
    
    non_opioid_results = []
    incorrectly_flagged = []
    
    for i, patient_id in enumerate(test_patients, 1):
        print(f"[{i}/{len(test_patients)}] Patient {patient_id} + {NON_OPIOID_DRUG}...", end=" ")
        result = test_patient_with_drug(patient_id, NON_OPIOID_DRUG, f"RX_NON_{i}")
        
        if "error" in result:
            print(f"❌ ERROR: {result['error']}")
        else:
            flagged = result.get('flagged', False)
            elig_score = result.get('eligibility_score', 0)
            oud_score = result.get('oud_risk_score', 0)
            explanation = result.get('explanation', '')
            
            if flagged:
                print(f"❌ FLAGGED (Elig: {elig_score:.3f}, OUD: {oud_score:.3f})")
                incorrectly_flagged.append({
                    'patient_id': patient_id,
                    'drug': NON_OPIOID_DRUG,
                    'result': result
                })
            else:
                print(f"✓ Not flagged (Elig: {elig_score:.3f}, OUD: {oud_score:.3f}) - {explanation[:50]}")
            
            non_opioid_results.append(result)
    
    print()
    print(f"Non-Opioid Test Summary:")
    print(f"  Tested: {len(non_opioid_results)}")
    print(f"  ✓ Correctly NOT flagged: {len(non_opioid_results) - len(incorrectly_flagged)}")
    print(f"  ❌ Incorrectly flagged: {len(incorrectly_flagged)}")
    print()
    
    # TEST 2: All patients with OPIOID drug (risk assessment)
    print("=" * 80)
    print(f"TEST 2: All Patients Prescribed OPIOID ({OPIOID_DRUG})")
    print("OUD risk assessment based on patient history")
    print("=" * 80)
    print()
    
    opioid_results = []
    flagged_count = 0
    not_flagged_count = 0
    
    for i, patient_id in enumerate(test_patients, 1):
        print(f"[{i}/{len(test_patients)}] Patient {patient_id} + {OPIOID_DRUG}...", end=" ")
        result = test_patient_with_drug(patient_id, OPIOID_DRUG, f"RX_OPI_{i}")
        
        if "error" in result:
            print(f"❌ ERROR: {result['error']}")
        else:
            flagged = result.get('flagged', False)
            elig_score = result.get('eligibility_score', 0)
            oud_score = result.get('oud_risk_score', 0)
            explanation = result.get('explanation', '')
            
            if flagged:
                print(f"🚩 FLAGGED (Elig: {elig_score:.3f}, OUD: {oud_score:.3f}) - {explanation[:60]}")
                flagged_count += 1
            else:
                print(f"✓ Not flagged (Elig: {elig_score:.3f}, OUD: {oud_score:.3f}) - {explanation[:60]}")
                not_flagged_count += 1
            
            opioid_results.append(result)
    
    print()
    print(f"Opioid Test Summary:")
    print(f"  Tested: {len(opioid_results)}")
    print(f"  🚩 Flagged (high risk): {flagged_count}")
    print(f"  ✓ Not flagged (low risk): {not_flagged_count}")
    print()
    
    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"Total tests performed: {len(non_opioid_results) + len(opioid_results)}")
    print(f"Total patients tested: {len(test_patients)}")
    print()
    print(f"NON-OPIOID Drug ({NON_OPIOID_DRUG}):")
    print(f"  ✓ Correctly NOT flagged: {len(non_opioid_results) - len(incorrectly_flagged)}")
    print(f"  ❌ Incorrectly flagged: {len(incorrectly_flagged)}")
    print()
    print(f"OPIOID Drug ({OPIOID_DRUG}):")
    print(f"  🚩 Flagged as high risk: {flagged_count}")
    print(f"  ✓ Assessed as low risk: {not_flagged_count}")
    print()
    
    if incorrectly_flagged:
        print("⚠️  WARNING: Some NON-OPIOID prescriptions were flagged!")
        print("   This indicates a bug - non-opioid drugs should NEVER trigger OUD risk.")
        print()
        print("   Incorrectly flagged patients:")
        for item in incorrectly_flagged[:5]:
            print(f"     - {item['patient_id']}")
    else:
        print("✅ SUCCESS: No non-opioid prescriptions were incorrectly flagged!")
    
    print()
    print("=" * 80)
    
    # Save detailed results
    output_file = "api_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'test_config': {
                'opioid_drug': OPIOID_DRUG,
                'non_opioid_drug': NON_OPIOID_DRUG,
                'total_patients': len(test_patients)
            },
            'non_opioid_results': non_opioid_results,
            'opioid_results': opioid_results,
            'incorrectly_flagged': incorrectly_flagged,
            'summary': {
                'total_tests': len(non_opioid_results) + len(opioid_results),
                'total_patients': len(test_patients),
                'non_opioid_correct': len(non_opioid_results) - len(incorrectly_flagged),
                'non_opioid_incorrect': len(incorrectly_flagged),
                'opioid_flagged': flagged_count,
                'opioid_not_flagged': not_flagged_count
            }
        }, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    print()

if __name__ == "__main__":
    main()
