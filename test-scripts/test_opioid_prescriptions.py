"""
Test Script: Send Opioid Prescriptions for All Patients and Collect AI Results

This script:
1. Connects to doctor_office database
2. Finds all patients (simulates opioid prescriptions)
3. Sends prescription requests to pharmacy API endpoint
4. Waits for AI processing to complete
5. Queries the pharmacy database for results
6. Exports results to CSV

Usage:
    python test_opioid_prescriptions.py
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import time
import uuid
import csv
from datetime import datetime
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configurations
DOCTOR_DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'rxaudit.postgres.database.azure.com'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': 'doctor_office',
    'user': os.getenv('DB_USER', 'posgres'),
    'password': os.getenv('DB_PASSWORD', 'UmaKiran12')
}

PHARMACY_DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'rxaudit.postgres.database.azure.com'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': 'autorxaudit',
    'user': os.getenv('DB_USER', 'posgres'),
    'password': os.getenv('DB_PASSWORD', 'UmaKiran12')
}

# API configuration
PHARMACY_API_URL = os.getenv('PHARMACY_API_URL', 'http://localhost:8004')

# Opioid drugs to test
OPIOID_DRUGS = [
    "Oxycodone 5mg",
    "Morphine 10mg",
    "Hydrocodone 5mg",
    "Fentanyl 25mcg",
    "Tramadol 50mg",
    "Codeine 30mg",
    "Hydromorphone 2mg"
]


def get_all_patients() -> List[Dict]:
    """Get all patients from doctor database."""
    print("Fetching patients from doctor_office database...")
    
    conn = psycopg2.connect(**DOCTOR_DB_CONFIG, cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT patient_id
        FROM patients
        ORDER BY patient_id
    """)
    
    patients = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"Found {len(patients)} patients")
    return [dict(p) for p in patients]


def send_prescription(patient_id: str, drug_name: str) -> Dict:
    """Send a prescription to pharmacy API endpoint."""
    prescription_uuid = str(uuid.uuid4())
    
    payload = {
        "patient_id": patient_id,
        "prescription_uuid": prescription_uuid,
        "drug_name": drug_name,
        "quantity": 30,
        "refills": 0,
        "encrypted": False  # Unencrypted for testing
    }
    
    try:
        response = requests.post(
            f"{PHARMACY_API_URL}/prescription",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "patient_id": patient_id,
                "prescription_uuid": prescription_uuid,
                "drug_name": drug_name,
                "message": response.json().get('message', 'Success')
            }
        else:
            return {
                "success": False,
                "patient_id": patient_id,
                "prescription_uuid": prescription_uuid,
                "drug_name": drug_name,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "patient_id": patient_id,
            "prescription_uuid": prescription_uuid,
            "drug_name": drug_name,
            "error": str(e)
        }


def send_all_prescriptions(patients: List[Dict]) -> List[Dict]:
    """Send prescriptions for all patients."""
    print("\nSending prescriptions to pharmacy API...")
    print("=" * 70)
    
    results = []
    
    for i, patient in enumerate(patients):
        patient_id = patient['patient_id']
        
        # Use a different opioid for each patient (cycle through the list)
        drug_name = OPIOID_DRUGS[i % len(OPIOID_DRUGS)]
        
        print(f"[{i+1}/{len(patients)}] Sending prescription for {patient_id} - {drug_name}...", end=" ")
        
        result = send_prescription(patient_id, drug_name)
        results.append(result)
        
        if result['success']:
            print("✓ Success")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    successful = sum(1 for r in results if r['success'])
    print("=" * 70)
    print(f"Sent {successful}/{len(results)} prescriptions successfully\n")
    
    return results


def wait_for_ai_processing(wait_seconds: int = 30):
    """Wait for AI processing to complete."""
    print(f"Waiting {wait_seconds} seconds for AI processing to complete...")
    
    for i in range(wait_seconds, 0, -1):
        print(f"  {i} seconds remaining...", end="\r")
        time.sleep(1)
    
    print("\n✓ Wait complete\n")


def get_ai_results() -> List[Dict]:
    """Query pharmacy database for AI results."""
    print("Querying pharmacy database for AI results...")
    
    conn = psycopg2.connect(**PHARMACY_DB_CONFIG, cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    
    # Query prescription_requests joined with prescription_review and patients
    cursor.execute("""
        SELECT 
            pr.prescription_id,
            pr.prescription_uuid,
            pr.patient_id,
            p.first_name || ' ' || p.last_name as patient_name,
            pr.drug_name,
            pr.quantity,
            pr.refills,
            pr.status as prescription_status,
            pr.prescribed_at,
            prv.id as review_id,
            prv.eligibility_score,
            prv.eligibility_prediction,
            prv.oud_risk_score,
            prv.oud_risk_prediction,
            prv.flagged,
            prv.flag_reason,
            prv.recommendation,
            prv.reviewed_by,
            prv.action,
            prv.action_reason,
            prv.created_at as review_created_at,
            prv.reviewed_at
        FROM prescription_requests pr
        LEFT JOIN prescription_review prv ON pr.prescription_id = prv.prescription_id
        LEFT JOIN patients p ON pr.patient_id = p.patient_id
        WHERE pr.prescribed_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
        ORDER BY pr.prescribed_at DESC
    """)
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"Found {len(results)} prescription records\n")
    return [dict(r) for r in results]


def export_to_csv(results: List[Dict], filename: str = None):
    """Export results to CSV file."""
    if not results:
        print("No results to export")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"opioid_prescriptions_test_{timestamp}.csv"
    
    # Define CSV columns
    fieldnames = [
        'prescription_id',
        'prescription_uuid',
        'patient_id',
        'patient_name',
        'drug_name',
        'quantity',
        'refills',
        'prescription_status',
        'prescribed_at',
        'review_id',
        'eligibility_score',
        'eligibility_prediction',
        'eligibility_label',
        'oud_risk_score',
        'oud_risk_prediction',
        'oud_risk_label',
        'flagged',
        'flag_reason',
        'recommendation',
        'reviewed_by',
        'action',
        'action_reason',
        'review_created_at',
        'reviewed_at'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Add human-readable labels
            row = result.copy()
            
            if result.get('eligibility_prediction') is not None:
                row['eligibility_label'] = 'Eligible' if result['eligibility_prediction'] == 1 else 'Not Eligible'
            else:
                row['eligibility_label'] = 'N/A'
            
            if result.get('oud_risk_prediction') is not None:
                row['oud_risk_label'] = 'High Risk' if result['oud_risk_prediction'] == 1 else 'Low Risk'
            else:
                row['oud_risk_label'] = 'N/A'
            
            writer.writerow(row)
    
    print(f"✓ Results exported to: {filename}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    total = len(results)
    processed = sum(1 for r in results if r.get('review_id') is not None)
    pending = total - processed
    
    print(f"Total Prescriptions: {total}")
    print(f"AI Processed: {processed} ({processed/total*100:.1f}%)" if total > 0 else "AI Processed: 0")
    print(f"Pending: {pending} ({pending/total*100:.1f}%)" if total > 0 else "Pending: 0")
    
    if processed > 0:
        flagged = sum(1 for r in results if r.get('flagged') is True)
        not_flagged = processed - flagged
        
        print(f"\nFlagged: {flagged} ({flagged/processed*100:.1f}%)")
        print(f"Not Flagged: {not_flagged} ({not_flagged/processed*100:.1f}%)")
        
        # Eligibility breakdown
        eligible = sum(1 for r in results if r.get('eligibility_prediction') == 1)
        not_eligible = sum(1 for r in results if r.get('eligibility_prediction') == 0)
        
        print(f"\nEligibility:")
        print(f"  Eligible: {eligible} ({eligible/processed*100:.1f}%)")
        print(f"  Not Eligible: {not_eligible} ({not_eligible/processed*100:.1f}%)")
        
        # OUD Risk breakdown
        high_risk = sum(1 for r in results if r.get('oud_risk_prediction') == 1)
        low_risk = sum(1 for r in results if r.get('oud_risk_prediction') == 0)
        
        print(f"\nOUD Risk:")
        print(f"  High Risk: {high_risk} ({high_risk/processed*100:.1f}%)")
        print(f"  Low Risk: {low_risk} ({low_risk/processed*100:.1f}%)")
        
        # Average scores
        avg_elig = sum(r.get('eligibility_score', 0) for r in results if r.get('eligibility_score') is not None) / processed
        avg_oud = sum(r.get('oud_risk_score', 0) for r in results if r.get('oud_risk_score') is not None) / processed
        
        print(f"\nAverage Scores:")
        print(f"  Eligibility: {avg_elig:.2%}")
        print(f"  OUD Risk: {avg_oud:.2%}")
    
    print("=" * 70 + "\n")


def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print("OPIOID PRESCRIPTION TEST SCRIPT")
    print("=" * 70 + "\n")
    
    try:
        # Step 1: Get all patients
        patients = get_all_patients()
        
        if not patients:
            print("No patients found in database. Exiting.")
            return
        
        # Optional: Limit number of patients for testing
        # patients = patients[:10]  # Uncomment to test with first 10 patients only
        
        # Step 2: Send prescriptions
        prescription_results = send_all_prescriptions(patients)
        
        # Step 3: Wait for AI processing
        wait_for_ai_processing(wait_seconds=45)  # Adjust wait time as needed
        
        # Step 4: Get AI results
        ai_results = get_ai_results()
        
        # Step 5: Export to CSV
        export_to_csv(ai_results)
        
        # Step 6: Print summary
        print_summary(ai_results)
        
        print("✓ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
