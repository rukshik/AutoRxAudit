import requests
import json

API_URL = "http://127.0.0.1:8000"

# Patients WITH pain diagnoses (should have higher eligibility scores)
ELIGIBLE_PATIENTS = [
    "20038695",
    "20033109"
]

# Patients WITHOUT pain diagnoses (from previous test - low eligibility)
INELIGIBLE_PATIENTS = [
    "20038478",
    "20002189"
]

print("=" * 80)
print("COMPARISON TEST: ELIGIBLE vs INELIGIBLE PATIENTS")
print("=" * 80)

print("\n[ELIGIBLE PATIENTS - Have pain diagnoses]")
print("Expected: Higher eligibility scores, potential approvals")
print("-" * 80)

for i, patient_id in enumerate(ELIGIBLE_PATIENTS, 1):
    payload = {
        "patient_id": patient_id,
        "drug_name": "Oxycodone 5mg",
        "prescriber_id": "DR001",
        "quantity": 30,
        "days_supply": 5
    }
    
    try:
        response = requests.post(f"{API_URL}/audit-prescription", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            flag_emoji = "🚩 FLAGGED" if result['flagged'] else "✅ APPROVED"
            
            print(f"\n{i}. Patient {patient_id} + Oxycodone... {flag_emoji}")
            print(f"   Eligibility: {result['eligibility_score']:.3f} (pred: {result['eligibility_prediction']})")
            print(f"   OUD Risk:    {result['oud_risk_score']:.3f} (pred: {result['oud_risk_prediction']})")
            print(f"   Reason: {result['flag_reason']}")
            
        else:
            print(f"\n{i}. Patient {patient_id}: API Error {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n{i}. Patient {patient_id}: Connection Error - {e}")

print("\n\n[INELIGIBLE PATIENTS - No pain diagnoses]")
print("Expected: Lower eligibility scores, all flagged")
print("-" * 80)

for i, patient_id in enumerate(INELIGIBLE_PATIENTS, 1):
    payload = {
        "patient_id": patient_id,
        "drug_name": "Oxycodone 5mg",
        "prescriber_id": "DR001",
        "quantity": 30,
        "days_supply": 5
    }
    
    try:
        response = requests.post(f"{API_URL}/audit-prescription", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            flag_emoji = "🚩 FLAGGED" if result['flagged'] else "✅ APPROVED"
            
            print(f"\n{i}. Patient {patient_id} + Oxycodone... {flag_emoji}")
            print(f"   Eligibility: {result['eligibility_score']:.3f} (pred: {result['eligibility_prediction']})")
            print(f"   OUD Risk:    {result['oud_risk_score']:.3f} (pred: {result['oud_risk_prediction']})")
            print(f"   Reason: {result['flag_reason']}")
            
        else:
            print(f"\n{i}. Patient {patient_id}: API Error {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n{i}. Patient {patient_id}: Connection Error - {e}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("- Eligible patients (with pain dx) should have pred=1 → potential approval")
print("- Ineligible patients (no pain dx) should have pred=0 → flagged")
print("- System is working if we see BOTH approvals AND flags")
print("=" * 80)
