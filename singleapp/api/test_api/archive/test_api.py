"""
Test script for AutoRxAudit API
Tests prescription auditing with various patient scenarios
"""

import requests
import json
from typing import Dict

API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test API health check."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    assert response.status_code == 200
    print("✓ Health check passed")


def test_audit_prescription(patient_id: str, description: str):
    """Test prescription audit for a patient."""
    print("\n" + "="*60)
    print(f"TEST: {description}")
    print("="*60)
    
    # Get patient data first
    print(f"\n1. Fetching patient {patient_id} data...")
    patient_response = requests.get(f"{API_BASE_URL}/patients/{patient_id}")
    
    if patient_response.status_code == 200:
        patient_data = patient_response.json()
        print(f"   Age: {patient_data['features'].get('age_at_first_admit', 'N/A'):.1f}")
        print(f"   Hospital admits: {patient_data['features'].get('n_hospital_admits', 0)}")
        print(f"   Opioid prescriptions: {patient_data['features'].get('opioid_rx_count', 0)}")
        print(f"   DRG severity: {patient_data['features'].get('avg_drg_severity', 0):.2f}")
    
    # Audit prescription
    print(f"\n2. Auditing prescription...")
    audit_payload = {
        "patient_id": patient_id,
        "prescription_id": f"RX_TEST_{patient_id}",
        "drug_name": "Oxycodone 10mg"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/audit-prescription",
        json=audit_payload
    )
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n3. Audit Result:")
        print(f"   {'='*56}")
        print(f"   Flagged: {'🚩 YES' if result['flagged'] else '✓ NO'}")
        print(f"   {'='*56}")
        print(f"   Eligibility Score: {result['eligibility_score']:.1%}")
        print(f"   Eligibility: {'ELIGIBLE' if result['eligibility_prediction'] == 1 else 'NOT ELIGIBLE'}")
        print(f"   OUD Risk Score: {result['oud_risk_score']:.1%}")
        print(f"   OUD Risk: {'HIGH RISK' if result['oud_risk_prediction'] == 1 else 'LOW RISK'}")
        print(f"   {'='*56}")
        print(f"   Reason: {result['flag_reason']}")
        print(f"   Recommendation: {result['recommendation']}")
        print(f"   {'='*56}")
        
        return result
    else:
        print(f"   Error: {response.json()}")
        return None


def test_audit_history(patient_id: str):
    """Test audit history retrieval."""
    print(f"\n4. Checking audit history for {patient_id}...")
    
    response = requests.get(f"{API_BASE_URL}/audit-history/{patient_id}")
    
    if response.status_code == 200:
        history = response.json()
        print(f"   Total audits: {history['total_audits']}")
        if history['audits']:
            latest = history['audits'][0]
            print(f"   Latest audit: {latest['audited_at']}")
            print(f"   Latest result: {'Flagged' if latest['flagged'] else 'Approved'}")


def test_stats():
    """Test statistics endpoint."""
    print("\n" + "="*60)
    print("TEST: Overall Statistics")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"\nTotal audits: {stats['total_audits']}")
        print(f"Flagged prescriptions: {stats['flagged_prescriptions']}")
        print(f"Flag rate: {stats['flag_rate']:.1%}")
        print(f"Avg eligibility score: {stats['avg_eligibility_score']:.1%}")
        print(f"Avg OUD risk score: {stats['avg_oud_risk_score']:.1%}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AUTORXAUDIT API TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Health check
        test_health_check()
        
        # Test 2-4: Audit different patient scenarios
        test_audit_prescription("PAT_00001", "Patient 1 - Mixed Risk Profile")
        
        test_audit_prescription("PAT_00050", "Patient 50 - Different Risk Profile")
        
        test_audit_prescription("PAT_00100", "Patient 100 - Another Scenario")
        
        # Test 5: Check audit history
        test_audit_history("PAT_00001")
        
        # Test 6: Overall statistics
        test_stats()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review audit decisions above")
        print("  2. Test with more patient IDs (PAT_00001 to PAT_00500)")
        print("  3. Check API docs at http://localhost:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("   Make sure the API is running: python app.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
