import requests
import json

API_URL = "http://127.0.0.1:8000"

# Test with eligible patients (have pain diagnosis + opioid history)
ELIGIBLE_PATIENTS = [
    "20000508", "20020815", "20021069", "20022949", "20025924"
]

print("=" * 80)
print("TESTING ELIGIBLE PATIENTS (should have high eligibility scores)")
print("Expected: Some should be APPROVED if OUD score is also low")
print("=" * 80)

for i, patient_id in enumerate(ELIGIBLE_PATIENTS, 1):
    payload = {
        "patient_id": patient_id,
        "drug_name": "Oxycodone 5mg",
        "prescriber_id": "DR001",
        "quantity": 30,
        "days_supply": 5
    }
    
    try:
        response = requests.post(f"{API_URL}/audit", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            flag_emoji = "🚩 FLAGGED" if result['flagged'] else "✅ APPROVED"
            
            print(f"\n[{i}/5] Patient {patient_id} + Oxycodone... {flag_emoji}")
            print(f"  Eligibility: {result['eligibility_score']:.3f} (pred: {result['eligibility_pred']})")
            print(f"  OUD Risk:    {result['oud_score']:.3f} (pred: {result['oud_pred']})")
            print(f"  Reason: {result['reason']}")
            
        else:
            print(f"\n[{i}/5] Patient {patient_id}: API Error {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n[{i}/5] Patient {patient_id}: Connection Error - {e}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("- If eligibility scores are HIGH (>0.5) → patient has pain diagnosis")
print("- If OUD scores are LOW (<0.5) → low addiction risk")
print("- Expected: Some APPROVED if both conditions met")
print("=" * 80)
