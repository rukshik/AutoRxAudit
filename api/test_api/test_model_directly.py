"""
Test the OUD model directly with patient data to see raw predictions
"""
import torch
import numpy as np
import pandas as pd
from feature_calculator import FeatureCalculator
import sys
sys.path.append('../ai-layer/model')
from dnn_models import DeepNet

# Database config
DB_CONFIG = {
    'host': 'autorxaudit-server.postgres.database.azure.com',
    'port': 5432,
    'database': 'mimiciv_demo_raw',
    'user': 'cloudsa',
    'password': 'Poornima@1985'
}

def test_model_predictions():
    """Test what model actually predicts for low-risk patients"""
    
    # Load OUD model
    MODEL_PATH = '../ai-layer/model/results/10000_v3/dnn_oud_risk_model.pth'
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    
    input_size = checkpoint['input_dim']  # Changed from 'input_size' to 'input_dim'
    model = DeepNet(input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler', None)
    feature_cols = checkpoint['feature_cols']
    
    print(f"Model expects {input_size} features")
    print(f"Feature columns: {feature_cols}")
    print()
    
    # Test with low-risk patients
    test_patients = ['20038478', '20002189', '20026527', '20045637', '20030188']
    
    calculator = FeatureCalculator(DB_CONFIG)
    
    for patient_id in test_patients:
        print(f"\n{'='*60}")
        print(f"Patient: {patient_id}")
        print(f"{'='*60}")
        
        # Calculate features
        features_dict = calculator.calculate_oud_features(patient_id)
        
        # Show key features
        print(f"  opioid_rx_count: {features_dict.get('opioid_rx_count', 0)}")
        print(f"  n_hospital_admits: {features_dict.get('n_hospital_admits', 0)}")
        print(f"  age: {features_dict.get('age_at_first_admit', 0)}")
        print(f"  avg_drg_severity: {features_dict.get('avg_drg_severity', 0)}")
        
        # Prepare features in correct order
        features_array = np.array([[features_dict[col] for col in feature_cols]], dtype=np.float32)
        
        print(f"\nRaw features shape: {features_array.shape}")
        print(f"Raw features (first 5): {features_array[0][:5]}")
        
        # Apply scaler
        if scaler is not None:
            features_scaled = scaler.transform(features_array)
            print(f"Scaled features (first 5): {features_scaled[0][:5]}")
        else:
            features_scaled = features_array
            print("WARNING: No scaler found!")
        
        # Run inference
        with torch.no_grad():
            x_tensor = torch.FloatTensor(features_scaled)
            logits = model(x_tensor).item()
            prob = 1 / (1 + np.exp(-logits))  # sigmoid
            prediction = 1 if prob >= 0.5 else 0
        
        print(f"\nModel output:")
        print(f"  Logits: {logits:.6f}")
        print(f"  Probability: {prob:.6f}")
        print(f"  Prediction: {prediction} ({'HIGH RISK' if prediction == 1 else 'LOW RISK'})")
        
        # Also check what the training data looked like for comparison
        

if __name__ == "__main__":
    test_model_predictions()
