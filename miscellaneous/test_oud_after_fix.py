"""
Test OUD risk model with patient 20000199 after data fix
"""
import requests
import json

# Test patient 20000199 (should be HIGH OUD risk)
patient_id = "20000199"
drug_name = "Oxycodone 5mg"

print("=" * 80)
print(f"Testing OUD Risk Model - Patient {patient_id}")
print("=" * 80)
print(f"Drug: {drug_name}")
print()

# Call API
url = "http://127.0.0.1:8000/audit-prescription"
payload = {
    "patient_id": patient_id,
    "drug_name": drug_name
}

print(f"POST {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = requests.post(url, json=payload, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ API Response:")
        print(f"  Flagged: {result['flagged']}")
        print(f"  Eligibility Score: {result['eligibility_score']:.3f} ({result['eligibility_score']*100:.1f}%)")
        print(f"  Eligibility Prediction: {result['eligibility_prediction']} ({'Eligible' if result['eligibility_prediction'] == 1 else 'Not Eligible'})")
        print(f"  OUD Risk Score: {result['oud_risk_score']:.3f} ({result['oud_risk_score']*100:.1f}%)")
        print(f"  OUD Risk Prediction: {result['oud_risk_prediction']} ({'HIGH RISK' if result['oud_risk_prediction'] == 1 else 'Low Risk'})")
        print(f"  Flag Reason: {result['flag_reason']}")
        print(f"  Recommendation: {result['recommendation']}")
        print()
        
        # Expected results
        print("=" * 80)
        print("EXPECTED vs ACTUAL:")
        print("=" * 80)
        print(f"Expected OUD Risk Score: ~0.91 (91% - HIGH RISK)")
        print(f"Actual OUD Risk Score: {result['oud_risk_score']:.3f} ({result['oud_risk_score']*100:.1f}%)")
        print()
        
        if result['oud_risk_score'] > 0.5:
            print("✅ SUCCESS: Model correctly identifies HIGH OUD RISK!")
            print(f"   Patient has 13 opioid prescriptions in database")
            print(f"   OUD risk score {result['oud_risk_score']:.3f} is consistent with training data")
        else:
            print("❌ FAILURE: Model still shows LOW OUD risk")
            print(f"   Expected score > 0.5, got {result['oud_risk_score']:.3f}")
            print(f"   This indicates feature calculation may still be incorrect")
        
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to API at http://127.0.0.1:8000")
    print("   Make sure the FastAPI server is running:")
    print("   cd api")
    print("   python app.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
