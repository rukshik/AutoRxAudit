"""Quick test to see what error is happening"""
import sys
sys.path.append('.')
from feature_calculator import FeatureCalculator

DB_CONFIG = {
    'host': 'autorxaudit-server.postgres.database.azure.com',
    'port': 5432,
    'database': 'mimiciv_demo_raw',
    'user': 'cloudsa',
    'password': 'Poornima@1985'
}

calc = FeatureCalculator(DB_CONFIG)
patient_id = '20038478'

print("Testing OUD feature calculation...")
try:
    features = calc.calculate_oud_features(patient_id)
    print(f"✓ Success! Got {len(features)} features")
    for k, v in features.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
