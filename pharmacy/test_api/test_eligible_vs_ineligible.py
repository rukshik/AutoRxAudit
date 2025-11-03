import uuid
import requests
import json

API_URL = "http://127.0.0.1:8004"

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
        "prescription_uuid": uuid.uuid4().hex,
        "drug_name": "Oxycodone 5mg",
        "prescriber_id": "345",
        "quantity": 30,
        "refills": 1
    }
    
    try:
        response = requests.post(f"{API_URL}/prescription", json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"\n{i}. Patient {patient_id} + Oxycodone... ", end="")            
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
        "prescription_uuid": uuid.uuid4().hex,
        "drug_name": "Oxycodone 5mg",
        "prescriber_id": "345",
        "quantity": 30,
        "refills": 1
    }
    
    try:
        response = requests.post(f"{API_URL}/prescription", json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"\n{i}. Patient {patient_id} + Oxycodone... ", end="")   
           
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
