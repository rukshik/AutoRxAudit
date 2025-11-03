"""
AutoRxAudit API - Opioid Prescription Auditing System

FastAPI application that:
1. Receives prescription requests
2. Queries PostgreSQL for patient features
3. Runs two models (Eligibility 50K + OUD Risk 10K)
4. Returns audit decision with explanation
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
from contextlib import contextmanager
from dotenv import load_dotenv
from feature_calculator import FeatureCalculator

# Load environment variables
load_dotenv()

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'autorxaudit'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

MODEL_DIR = '../ai-layer/model'
ELIGIBILITY_MODEL_PATH = os.path.join(MODEL_DIR, 'results/50000_v3/dnn_eligibility_model.pth')
OUD_MODEL_PATH = os.path.join(MODEL_DIR, 'results/10000_v3/dnn_oud_risk_model.pth')

# Feature lists will be loaded from model checkpoints during model loading
# DO NOT hardcode these - they MUST match exactly what the models were trained with
ELIGIBILITY_FEATURES = None  # Will be set by load_models()
OUD_FEATURES = None  # Will be set by load_models()

# Initialize FastAPI app
app = FastAPI(
    title="AutoRxAudit API",
    description="AI-powered opioid prescription auditing system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PrescriptionRequest(BaseModel):
    """Prescription audit request."""
    patient_id: str = Field(..., description="Patient identifier")
    prescription_id: Optional[str] = Field(None, description="Prescription identifier")
    drug_name: Optional[str] = Field(None, description="Drug name (for logging)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PAT_00001",
                "prescription_id": "RX_12345",
                "drug_name": "Oxycodone 5mg"
            }
        }


class AuditResponse(BaseModel):
    """Audit decision response."""
    flagged: bool = Field(..., description="Whether prescription should be flagged")
    patient_id: str
    prescription_id: Optional[str]
    
    # Model predictions
    eligibility_score: float = Field(..., description="Eligibility probability (0-1)")
    eligibility_prediction: int = Field(..., description="1=Eligible, 0=Not Eligible")
    oud_risk_score: float = Field(..., description="OUD risk probability (0-1)")
    oud_risk_prediction: int = Field(..., description="1=High Risk, 0=Low Risk")
    
    # Explanation
    flag_reason: str = Field(..., description="Reason for flagging")
    recommendation: str = Field(..., description="Recommended action")
    
    # Metadata
    audited_at: str


class PatientFeaturesResponse(BaseModel):
    """Patient feature data response."""
    patient_id: str
    features: Dict[str, float]
    created_at: str


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@contextmanager
def get_db():
    """Database connection context manager."""
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


# ============================================================================
# DEEP NEURAL NETWORK MODEL
# ============================================================================

class DeepNet(nn.Module):
    """Deep Neural Network with BatchNorm - matches training architecture."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.3):
        super(DeepNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with BatchNorm
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no sigmoid - use BCEWithLogitsLoss during training)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load trained models from checkpoints and extract feature lists."""
    global ELIGIBILITY_FEATURES, OUD_FEATURES
    
    print("Loading models...")
    
    # Load Eligibility model checkpoint
    eligibility_checkpoint = torch.load(ELIGIBILITY_MODEL_PATH, map_location='cpu', weights_only=False)
    ELIGIBILITY_FEATURES = eligibility_checkpoint['feature_cols']  # Get correct feature order from checkpoint
    eligibility_input_dim = eligibility_checkpoint.get('input_dim', len(ELIGIBILITY_FEATURES))
    
    eligibility_model = DeepNet(input_dim=eligibility_input_dim)
    eligibility_model.load_state_dict(eligibility_checkpoint['model_state_dict'])
    eligibility_model.eval()
    eligibility_scaler = eligibility_checkpoint.get('scaler', None)
    
    print(f"  Eligibility model: {eligibility_input_dim} features")
    print(f"  Features: {ELIGIBILITY_FEATURES}")
    
    # Load OUD Risk model checkpoint
    oud_checkpoint = torch.load(OUD_MODEL_PATH, map_location='cpu', weights_only=False)
    OUD_FEATURES = oud_checkpoint['feature_cols']  # Get correct feature order from checkpoint
    oud_input_dim = oud_checkpoint.get('input_dim', len(OUD_FEATURES))
    
    oud_model = DeepNet(input_dim=oud_input_dim)
    oud_model.load_state_dict(oud_checkpoint['model_state_dict'])
    oud_model.eval()
    oud_scaler = oud_checkpoint.get('scaler', None)
    
    print(f"  OUD model: {oud_input_dim} features")
    print(f"  Features: {OUD_FEATURES}")
    
    print("✓ Models loaded successfully")
    return eligibility_model, eligibility_scaler, oud_model, oud_scaler


# Global model instances
eligibility_model, eligibility_scaler, oud_model, oud_scaler = load_models()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_patient_features(conn, patient_id: str) -> Dict:
    """Fetch patient features from database."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM patients WHERE patient_id = %s
    """, (patient_id,))
    
    patient = cursor.fetchone()
    cursor.close()
    
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    return dict(patient)


def prepare_features(patient_data: Dict, feature_list: List[str]) -> np.ndarray:
    """Extract and order features for model input."""
    features = []
    for feature in feature_list:
        value = patient_data.get(feature, 0)
        # Convert boolean to int
        if isinstance(value, bool):
            value = int(value)
        # Handle None
        if value is None:
            value = 0
        features.append(float(value))
    
    return np.array(features, dtype=np.float32).reshape(1, -1)


def run_inference(model, features: np.ndarray) -> tuple:
    """Run model inference and return probability."""
    with torch.no_grad():
        x_tensor = torch.tensor(features, dtype=torch.float32)
        logits = model(x_tensor).item()
        # Apply sigmoid to convert logits to probability
        prob = 1 / (1 + np.exp(-logits))
        prediction = 1 if prob >= 0.5 else 0
    
    return prob, prediction


def log_audit(conn, audit_data: Dict):
    """Log audit decision to database."""
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO audit_logs (
            patient_id, prescription_id, flagged,
            eligibility_score, eligibility_prediction,
            oud_risk_score, oud_risk_prediction,
            flag_reason, calculated_features
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
    """, (
        audit_data['patient_id'],
        audit_data['prescription_id'],
        audit_data['flagged'],
        audit_data['eligibility_score'],
        audit_data['eligibility_prediction'],
        audit_data['oud_risk_score'],
        audit_data['oud_risk_prediction'],
        audit_data['flag_reason'],
        audit_data.get('calculated_features', '{}')
    ))
    
    conn.commit()
    cursor.close()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "healthy",
        "service": "AutoRxAudit API",
        "version": "1.0.0",
        "models": {
            "eligibility": "DNN 50K (16 features)",
            "oud_risk": "DNN 10K (19 features)"
        }
    }


@app.post("/audit-prescription", response_model=AuditResponse)
async def audit_prescription(request: PrescriptionRequest):
    """
    Audit a prescription request.
    
    Returns audit decision based on:
    - Drug Type: Is this an opioid? (Non-opioids auto-approved)
    - Eligibility Model (50K): Does patient have clinical need?
    - OUD Risk Model (10K): Is patient at high risk for OUD?
    
    Flags prescription if it's an OPIOID AND EITHER:
    - Patient is NOT eligible (no clinical need), OR
    - Patient has HIGH OUD risk
    """
    
    # 1. Check if drug is an opioid
    opioid_keywords = [
        'fentanyl', 'morphine', 'oxycodone', 'hydrocodone', 'hydromorphone',
        'oxymorphone', 'codeine', 'tramadol', 'methadone', 'buprenorphine',
        'meperidine', 'tapentadol', 'opioid', 'narcotic'
    ]
    
    print(f"DEBUG: Received drug_name='{request.drug_name}'")
    drug_name = (request.drug_name or "").lower()
    print(f"DEBUG: Lowercase drug_name='{drug_name}'")
    is_opioid = any(keyword in drug_name for keyword in opioid_keywords)
    print(f"DEBUG: is_opioid={is_opioid}")
    
    # 2. If NOT an opioid, auto-approve (no OUD risk)
    if not is_opioid:
        print(f"✓ Non-opioid drug '{request.drug_name}' - auto-approved (no risk assessment)")
        return AuditResponse(
            flagged=False,
            patient_id=request.patient_id,
            prescription_id=request.prescription_id or f"RX_{request.patient_id}",
            eligibility_score=1.0,
            eligibility_prediction=1,
            oud_risk_score=0.0,
            oud_risk_prediction=0,
            flag_reason="Non-opioid prescription - no OUD risk assessment required",
            recommendation="APPROVED: Non-opioid medication",
            audited_at=datetime.now().isoformat()
        )
    
    # 3. For OPIOIDS, perform full risk assessment
    print(f"Opioid drug '{request.drug_name}' detected - performing risk assessment...")
    
    try:
        # 4. Initialize feature calculator
        calculator = FeatureCalculator(DB_CONFIG)
        
        # 5. Calculate features dynamically from raw EHR data
        print(f"Calculating features for patient {request.patient_id}...")
        eligibility_features_dict = calculator.calculate_eligibility_features(request.patient_id)
        oud_features_dict = calculator.calculate_oud_features(request.patient_id)
    except Exception as e:
        print(f"ERROR calculating features: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature calculation failed: {str(e)}")
    
    # 6. Prepare feature arrays for models
    eligibility_features = np.array([eligibility_features_dict[f] for f in ELIGIBILITY_FEATURES], dtype=np.float32).reshape(1, -1)
    oud_features = np.array([oud_features_dict[f] for f in OUD_FEATURES], dtype=np.float32).reshape(1, -1)
    
    # 7. Apply scalers (normalize features)
    if eligibility_scaler is not None:
        eligibility_features = eligibility_scaler.transform(eligibility_features)
    if oud_scaler is not None:
        oud_features = oud_scaler.transform(oud_features)
    
    # 8. Run inference
    eligibility_score, eligibility_pred = run_inference(eligibility_model, eligibility_features)
    oud_score, oud_pred = run_inference(oud_model, oud_features)
    
    with get_db() as conn:
        
        # 9. Determine if prescription should be flagged
        # Flag if: NOT eligible (pred=0) OR high OUD risk (pred=1)
        not_eligible = (eligibility_pred == 0)
        high_oud_risk = (oud_pred == 1)
        flagged = not_eligible or high_oud_risk
        
        # 10. Generate explanation
        reasons = []
        if not_eligible:
            reasons.append(f"No clinical need for opioids (eligibility: {eligibility_score:.1%})")
        if high_oud_risk:
            reasons.append(f"High OUD risk detected (risk: {oud_score:.1%})")
        
        if flagged:
            flag_reason = " AND ".join(reasons)
            recommendation = "REVIEW REQUIRED: Consider alternative pain management or additional evaluation"
        else:
            flag_reason = "Patient eligible with acceptable OUD risk"
            recommendation = "APPROVED: Opioid prescription meets clinical criteria"
        
        # 11. Log audit decision with calculated features
        import json
        all_features = {**eligibility_features_dict, **oud_features_dict}
        
        audit_data = {
            'patient_id': request.patient_id,
            'prescription_id': request.prescription_id,
            'flagged': flagged,
            'eligibility_score': eligibility_score,
            'eligibility_prediction': eligibility_pred,
            'oud_risk_score': oud_score,
            'oud_risk_prediction': oud_pred,
            'flag_reason': flag_reason,
            'calculated_features': json.dumps(all_features)
        }
        log_audit(conn, audit_data)
        
        # 12. Return response
        return AuditResponse(
            flagged=flagged,
            patient_id=request.patient_id,
            prescription_id=request.prescription_id,
            eligibility_score=eligibility_score,
            eligibility_prediction=eligibility_pred,
            oud_risk_score=oud_score,
            oud_risk_prediction=oud_pred,
            flag_reason=flag_reason,
            recommendation=recommendation,
            audited_at=datetime.now().isoformat()
        )


@app.get("/patients/{patient_id}", response_model=PatientFeaturesResponse)
async def get_patient(patient_id: str):
    """Get patient feature data."""
    with get_db() as conn:
        patient_data = get_patient_features(conn, patient_id)
        
        # Remove metadata fields
        features = {k: v for k, v in patient_data.items() 
                   if k not in ['patient_id', 'created_at', 'updated_at']}
        
        return PatientFeaturesResponse(
            patient_id=patient_id,
            features=features,
            created_at=patient_data.get('created_at', '').isoformat() if patient_data.get('created_at') else ''
        )


@app.get("/audit-history/{patient_id}")
async def get_audit_history(patient_id: str, limit: int = 10):
    """Get audit history for a patient."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM audit_logs 
            WHERE patient_id = %s 
            ORDER BY audited_at DESC 
            LIMIT %s
        """, (patient_id, limit))
        
        history = cursor.fetchall()
        cursor.close()
        
        return {
            "patient_id": patient_id,
            "total_audits": len(history),
            "audits": [dict(row) for row in history]
        }


@app.get("/stats")
async def get_stats():
    """Get audit statistics."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_audits,
                SUM(CASE WHEN flagged THEN 1 ELSE 0 END) as flagged_count,
                AVG(eligibility_score) as avg_eligibility_score,
                AVG(oud_risk_score) as avg_oud_risk_score
            FROM audit_logs
        """)
        
        stats = cursor.fetchone()
        cursor.close()
        
        return {
            "total_audits": stats['total_audits'] or 0,
            "flagged_prescriptions": stats['flagged_count'] or 0,
            "flag_rate": (stats['flagged_count'] or 0) / max(stats['total_audits'] or 1, 1),
            "avg_eligibility_score": float(stats['avg_eligibility_score'] or 0),
            "avg_oud_risk_score": float(stats['avg_oud_risk_score'] or 0)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
