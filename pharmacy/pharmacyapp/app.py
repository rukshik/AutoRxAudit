"""
AutoRxAudit API - Opioid Prescription Auditing System

FastAPI application that:
1. Receives prescription requests
2. Queries PostgreSQL for patient features
3. Runs two models (Eligibility 50K + OUD Risk 10K)
4. Returns audit decision with explanation
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
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
import httpx
import asyncio

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

# Blockchain service configuration
BLOCKCHAIN_SERVICE_URL = os.getenv('BLOCKCHAIN_SERVICE_URL', 'http://localhost:8001')

# Doctor API configuration
DOCTOR_API_URL = os.getenv('DOCTOR_API_URL', 'http://localhost:8003')

MODEL_DIR = '../../ai-layer/model'
ELIGIBILITY_MODEL_PATH = os.path.join(MODEL_DIR, 'results/50000_v3/dnn_eligibility_model.pth')
OUD_MODEL_PATH = os.path.join(MODEL_DIR, 'results/10000_v3/dnn_oud_risk_model.pth')

# Feature lists will be loaded from model checkpoints during model loading
# DO NOT hardcode these - they MUST match exactly what the models were trained with
ELIGIBILITY_FEATURES = None  # Will be set by load_models()
OUD_FEATURES = None  # Will be set by load_models()

# Initialize FastAPI app
app = FastAPI(
    title="AutoRxAudit Pharmacy",
    description="AI-powered opioid prescription auditing system - Pharmacy Portal",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory="templates")

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
    """Prescription request from doctor."""
    patient_id: str = Field(..., description="Patient identifier")
    prescription_uuid: str = Field(..., description="Prescription UUID from doctor system")
    drug_name: str = Field(..., description="Drug name with dosage")
    quantity: Optional[int] = Field(default=None, description="Quantity prescribed")
    refills: Optional[int] = Field(default=None, description="Number of refills")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PAT_00001",
                "prescription_uuid": "550e8400-e29b-41d4-a716-446655440000",
                "drug_name": "Oxycodone 5mg",
                "quantity": 30,
                "refills": 0
            }
        }


class PrescriptionReceiptResponse(BaseModel):
    """Immediate response when prescription is received."""
    status: str = Field(default="success", description="Receipt status")
    prescription_uuid: str = Field(..., description="Prescription UUID")
    message: str = Field(default="Prescription received and queued for review", description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "prescription_uuid": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Prescription received and queued for review"
            }
        }


class AuditResponse(BaseModel):
    """Audit decision response."""
    flagged: bool = Field(..., description="Whether prescription should be flagged")
    patient_id: str
    prescription_id: Optional[str]
    audit_id: Optional[int] = None
    
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


class LoginRequest(BaseModel):
    """User login request."""
    email: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""
    success: bool
    user: Dict


class AuditActionRequest(BaseModel):
    """Pharmacist action on audit result."""
    audit_id: int  # Required - audit log ID to update
    action: str  # 'APPROVED', 'DENIED', 'OVERRIDE_APPROVE', 'OVERRIDE_DENY'
    action_reason: Optional[str] = None


class AuditActionResponse(BaseModel):
    """Audit action response."""
    action_id: int
    success: bool
    message: str


class AuditHistoryItem(BaseModel):
    """Audit history item."""
    audit_id: int  # Changed from action_id to audit_id
    patient_id: str
    drug_name: str
    user_email: Optional[str] = None  # Optional - None if not reviewed yet
    user_name: Optional[str] = None  # Optional - None if not reviewed yet
    flagged: bool
    eligibility_score: float
    oud_risk_score: float
    action: Optional[str] = None  # Optional - None if not reviewed yet
    action_reason: Optional[str] = None
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
# BACKGROUND AI PROCESSING
# ============================================================================

async def process_prescription_ai_check(prescription_id: int, patient_id: str, drug_name: str, prescription_uuid: str):
    """
    Background task to run AI check on prescription.
    Updates prescription_review table with results and updates prescription status.
    """
    try:
        print(f"Starting AI check for prescription_id={prescription_id}, uuid={prescription_uuid}")
        
        # Check if drug is an opioid
        opioid_keywords = [
            'fentanyl', 'morphine', 'oxycodone', 'hydrocodone', 'hydromorphone',
            'oxymorphone', 'codeine', 'tramadol', 'methadone', 'buprenorphine',
            'meperidine', 'tapentadol', 'opioid', 'narcotic'
        ]
        
        drug_name_lower = drug_name.lower()
        is_opioid = any(keyword in drug_name_lower for keyword in opioid_keywords)
        
        # Initialize feature calculator
        calculator = FeatureCalculator(DB_CONFIG)
        
        # Calculate features
        eligibility_features_dict = calculator.calculate_eligibility_features(patient_id)
        oud_features_dict = calculator.calculate_oud_features(patient_id)
        
        # Prepare feature arrays
        eligibility_features = np.array([eligibility_features_dict[f] for f in ELIGIBILITY_FEATURES], dtype=np.float32).reshape(1, -1)
        oud_features = np.array([oud_features_dict[f] for f in OUD_FEATURES], dtype=np.float32).reshape(1, -1)
        
        # Apply scalers
        if eligibility_scaler is not None:
            eligibility_features = eligibility_scaler.transform(eligibility_features)
        if oud_scaler is not None:
            oud_features = oud_scaler.transform(oud_features)
        
        # Run inference
        eligibility_score, eligibility_pred = run_inference(eligibility_model, eligibility_features)
        oud_score, oud_pred = run_inference(oud_model, oud_features)
        
        # Determine if flagged
        not_eligible = (eligibility_pred == 0)
        high_oud_risk = (oud_pred == 1)
        flagged = not_eligible or high_oud_risk
        
        # Generate explanation
        reasons = []
        if not_eligible:
            reasons.append(f"No clinical need for opioids (eligibility: {eligibility_score:.1%})")
        if high_oud_risk:
            reasons.append(f"High OUD risk detected (risk: {oud_score:.1%})")
        
        if flagged:
            flag_reason = " AND ".join(reasons)
            recommendation = "REVIEW REQUIRED: Consider alternative pain management or additional evaluation"
            new_status = 'AI_FLAGGED'
        else:
            flag_reason = "Patient eligible with acceptable OUD risk"
            recommendation = "AI APPROVED: Opioid prescription meets clinical criteria"
            new_status = 'AI_APPROVED'
        
        # Save to database
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Insert into prescription_review table
            cursor.execute("""
                INSERT INTO prescription_review (
                    prescription_id, eligibility_score, eligibility_prediction,
                    oud_risk_score, oud_risk_prediction,
                    flagged, flag_reason, recommendation
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                prescription_id, eligibility_score, eligibility_pred,
                oud_score, oud_pred,
                flagged, flag_reason, recommendation
            ))
            review_id = cursor.fetchone()['id']
            
            # Update prescription status
            cursor.execute("""
                UPDATE prescription_requests
                SET status = %s
                WHERE prescription_id = %s
            """, (new_status, prescription_id))
            
            conn.commit()
            cursor.close()
            
            print(f"✓ AI check completed: prescription_id={prescription_id}, status={new_status}, review_id={review_id}")
            
            # Log AI review to blockchain
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    blockchain_response = await client.post(
                        "http://localhost:8001/pharmacy/ai-review",
                        json={
                            "prescription_uuid": prescription_uuid,
                            "flagged": bool(flagged),
                            "eligibility_score": int(eligibility_score * 100),
                            "oud_risk_score": int(oud_score * 100),
                            "flag_reason": flag_reason,
                            "recommendation": recommendation
                        }
                    )
                    if blockchain_response.status_code == 200:
                        print(f"✓ AI review logged to blockchain for {prescription_uuid}")
                    else:
                        print(f"⚠️  Blockchain AI logging failed: {blockchain_response.status_code}")
            except Exception as bc_err:
                print(f"⚠️  Blockchain AI logging error: {bc_err}")
            
    except Exception as e:
        print(f"ERROR in AI check for prescription_id={prescription_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Update status to AI_ERROR
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE prescription_requests
                    SET status = 'AI_ERROR'
                    WHERE prescription_id = %s
                """, (prescription_id,))
                conn.commit()
                cursor.close()
        except Exception as db_error:
            print(f"ERROR updating status to AI_ERROR: {db_error}")


# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

# Simple session storage (in-memory - for production use Redis or database)
active_sessions = {}


def create_session(email: str) -> str:
    """Create a simple session ID."""
    import uuid
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = email
    return session_id


def get_user_from_session(session_id: str) -> Optional[Dict]:
    """Get user from session ID."""
    email = active_sessions.get(session_id)
    if not email:
        return None
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, email, full_name, role FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        return dict(user) if user else None


async def get_current_user(session_id: str = Depends(lambda: None)):
    """Get current user from session (simplified - no auth for now)."""
    # For development: skip authentication
    # In production, check session_id from cookie/header
    return {"user_id": 1, "email": "test@test.com", "full_name": "Test User", "role": "doctor"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def send_audit_to_blockchain(audit_data: Dict) -> Optional[Dict]:
    """Send audit record to blockchain service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{BLOCKCHAIN_SERVICE_URL}/record-audit",
                json={
                    "audit_id": audit_data['audit_id'],
                    "prescription_id": audit_data['prescription_id'],
                    "patient_id": audit_data['patient_id'],
                    "drug_name": audit_data['drug_name'],
                    "eligibility_score": int(audit_data['eligibility_score'] * 100),
                    "eligibility_prediction": audit_data['eligibility_prediction'],
                    "oud_risk_score": int(audit_data['oud_risk_score'] * 100),
                    "oud_risk_prediction": audit_data['oud_risk_prediction'],
                    "flagged": audit_data['flagged'],
                    "flag_reason": audit_data['flag_reason'],
                    "recommendation": audit_data['recommendation']
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Blockchain recording failed: {response.text}")
                return None
    except Exception as e:
        print(f"⚠️  Blockchain service unavailable: {e}")
        return None


async def send_pharmacist_action_to_blockchain(action_data: Dict) -> Optional[Dict]:
    """Send pharmacist action to blockchain service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{BLOCKCHAIN_SERVICE_URL}/record-pharmacist-action",
                json={
                    "audit_id": action_data['audit_id'],
                    "action": action_data['action'],
                    "action_reason": action_data['action_reason'],
                    "reviewed_by": str(action_data['reviewed_by']),
                    "reviewed_by_name": action_data['reviewed_by_name'],
                    "reviewed_by_email": action_data['reviewed_by_email']
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Blockchain recording failed: {response.text}")
                return None
    except Exception as e:
        print(f"⚠️  Blockchain service unavailable: {e}")
        return None


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

@app.get("/api/health")
async def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "service": "AutoRxAudit Pharmacy",
        "version": "1.0.0",
        "models": {
            "eligibility": "DNN 50K (16 features)",
            "oud_risk": "DNN 10K (19 features)"
        }
    }


@app.post("/prescription", response_model=PrescriptionReceiptResponse)
async def receive_prescription(request: PrescriptionRequest):
    """
    Receive a prescription from doctor.
    
    Workflow:
    1. Store prescription immediately with status='RECEIVED'
    2. Return 200 OK immediately
    3. Kick off AI check in background
    4. Background task updates status to AI_APPROVED, AI_FLAGGED, or AI_ERROR
    """
    
    print(f"Received prescription: uuid={request.prescription_uuid}, patient={request.patient_id}, drug={request.drug_name}")
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Insert prescription with RECEIVED status
            cursor.execute("""
                INSERT INTO prescription_requests (
                    patient_id, drug_name, prescription_uuid, 
                    quantity, refills, status
                ) VALUES (%s, %s, %s, %s, %s, 'RECEIVED')
                RETURNING prescription_id
            """, (
                request.patient_id, 
                request.drug_name, 
                request.prescription_uuid,
                request.quantity,
                request.refills
            ))
            
            prescription_id = cursor.fetchone()['prescription_id']
            conn.commit()
            cursor.close()
            
            print(f"✓ Prescription stored: prescription_id={prescription_id}, status=RECEIVED")
            
            # Kick off AI check in background (fire and forget)
            asyncio.create_task(
                process_prescription_ai_check(
                    prescription_id=prescription_id,
                    patient_id=request.patient_id,
                    drug_name=request.drug_name,
                    prescription_uuid=request.prescription_uuid
                )
            )
            
            # Return immediate success response
            return PrescriptionReceiptResponse(
                status="success",
                prescription_uuid=request.prescription_uuid,
                message="Prescription received and queued for review"
            )
            
    except Exception as e:
        print(f"ERROR storing prescription: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to store prescription: {str(e)}")


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


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """User login endpoint - simple password matching."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, email, password, full_name, role FROM users WHERE email = %s AND is_active = TRUE",
            (request.email,)
        )
        user = cursor.fetchone()
        
        if not user or user['password'] != request.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Update last login
        cursor.execute("UPDATE users SET last_login = NOW() WHERE user_id = %s", (user['user_id'],))
        conn.commit()
        cursor.close()
        
        # Create simple session
        session_id = create_session(user['email'])
        
        return {
            "success": True,
            "user": {
                "user_id": user['user_id'],
                "email": user['email'],
                "full_name": user['full_name'],
                "role": user['role'],
                "session_id": session_id
            }
        }


@app.get("/api/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return current_user


@app.get("/api/patients")
async def get_patients(current_user: dict = Depends(get_current_user)):
    """Get list of patients for dropdown."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_id, date_of_birth, gender
            FROM patients
            ORDER BY patient_id
            LIMIT 100
        """)
        patients = cursor.fetchall()
        cursor.close()
        return [dict(p) for p in patients]


@app.get("/api/drugs")
async def get_drugs(current_user: dict = Depends(get_current_user)):
    """Get list of drugs (opioids and non-opioids) for dropdown."""
    return {
        "opioids": [
            "Oxycodone 5mg",
            "Morphine 10mg",
            "Hydrocodone 5mg",
            "Fentanyl 25mcg",
            "Hydromorphone 2mg",
            "Codeine 30mg",
            "Tramadol 50mg"
        ],
        "non_opioids": [
            "Acetaminophen 500mg",
            "Ibuprofen 400mg",
            "Naproxen 500mg",
            "Aspirin 325mg",
            "Celecoxib 200mg"
        ]
    }


# ============================================================================
# AUDIT ACTION ENDPOINTS
# ============================================================================

@app.post("/api/audit-action", response_model=AuditActionResponse)
async def record_audit_action(
    request: AuditActionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Record pharmacist decision on prescription audit (approve/deny/override)."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Update audit_logs with pharmacist decision
        cursor.execute("""
            UPDATE audit_logs
            SET reviewed_by = %s,
                action = %s,
                action_reason = %s,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE audit_id = %s
            RETURNING audit_id
        """, (
            current_user['user_id'],
            request.action,
            request.action_reason,
            request.audit_id
        ))
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            raise HTTPException(status_code=404, detail=f"Audit ID {request.audit_id} not found")
        
        audit_id = result['audit_id']
        
        # Update prescription status based on action
        status_map = {
            'APPROVED': 'APPROVED',
            'OVERRIDE_APPROVE': 'APPROVED',
            'DENIED': 'DENIED',
            'OVERRIDE_DENY': 'DENIED'
        }
        new_status = status_map.get(request.action, 'PENDING')
        
        cursor.execute("""
            UPDATE prescription_requests
            SET status = %s
            WHERE prescription_id = (
                SELECT prescription_id FROM audit_logs WHERE audit_id = %s
            )
        """, (new_status, audit_id))
        
        conn.commit()
        cursor.close()
        
        # Send pharmacist action to blockchain (async, non-blocking)
        blockchain_result = await send_pharmacist_action_to_blockchain({
            'audit_id': audit_id,
            'action': request.action,
            'action_reason': request.action_reason or '',
            'reviewed_by': current_user['user_id'],
            'reviewed_by_name': current_user.get('full_name', ''),
            'reviewed_by_email': current_user.get('email', '')
        })
        
        if blockchain_result:
            print(f"✓ Pharmacist action recorded on blockchain: {blockchain_result.get('transaction_hash', 'N/A')}")
        
        return {
            "action_id": audit_id,
            "success": True,
            "message": f"Pharmacist decision '{request.action}' recorded successfully"
        }


@app.get("/api/audit-history", response_model=List[AuditHistoryItem])
async def get_audit_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get audit history with user actions."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                al.audit_id,
                al.prescription_id,
                al.patient_id,
                al.drug_name,
                al.eligibility_score,
                al.eligibility_prediction,
                al.oud_risk_score,
                al.oud_risk_prediction,
                al.flagged,
                al.flag_reason,
                al.recommendation,
                al.audited_at,
                al.reviewed_by,
                al.action,
                al.action_reason,
                al.reviewed_at,
                u.full_name as clinician_name,
                u.email as clinician_email,
                u.role as clinician_role
            FROM audit_logs al
            LEFT JOIN users u ON al.reviewed_by = u.user_id
            ORDER BY al.audited_at DESC
            LIMIT %s
        """, (limit,))
        
        audits = cursor.fetchall()
        cursor.close()
        
        return [
            {
                "audit_id": a['audit_id'],
                "patient_id": a['patient_id'],
                "drug_name": a['drug_name'],
                "user_email": a['clinician_email'] if a['clinician_email'] else None,
                "user_name": a['clinician_name'] if a['clinician_name'] else None,
                "flagged": a['flagged'],
                "eligibility_score": float(a['eligibility_score']),
                "oud_risk_score": float(a['oud_risk_score']),
                "action": a['action'] if a['action'] else None,
                "action_reason": a['action_reason'],
                "created_at": a['audited_at'].isoformat() if a['audited_at'] else None
            }
            for a in audits
        ]


@app.get("/api/blockchain-audits")
async def get_blockchain_audits(limit: int = 50, current_user: dict = Depends(get_current_user)):
    """Get all audit records from blockchain."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BLOCKCHAIN_SERVICE_URL}/audit-records/all?limit={limit}")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Blockchain service not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch blockchain records: {str(e)}")


@app.get("/api/blockchain-audit/{blockchain_id}")
async def get_blockchain_audit(blockchain_id: int, current_user: dict = Depends(get_current_user)):
    """Get specific audit record from blockchain by blockchain ID."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BLOCKCHAIN_SERVICE_URL}/audit-record/{blockchain_id}")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="Blockchain record not found")
            else:
                raise HTTPException(status_code=503, detail="Blockchain service unavailable")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Blockchain service not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch blockchain record: {str(e)}")


# ============================================================================
# WEB UI ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Redirect to login page."""
    return RedirectResponse(url="/pharmacy/login")


@app.get("/pharmacy/login", response_class=HTMLResponse)
def login_page(request: Request):
    """Pharmacy login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/pharmacy/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    """Pharmacy login endpoint."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, email, password, full_name, role FROM users WHERE email = %s AND is_active = TRUE",
            (email,)
        )
        user = cursor.fetchone()
        cursor.close()
        
        if not user or user['password'] != password:
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "Invalid email or password"
            })
        
        # Redirect to prescriptions list
        return RedirectResponse(
            url=f"/pharmacy/prescriptions?user_id={user['user_id']}",
            status_code=303
        )


@app.get("/pharmacy/prescriptions", response_class=HTMLResponse)
def prescriptions_page(request: Request, user_id: int):
    """Prescription queue page."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get user info
        cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return RedirectResponse(url="/pharmacy/login")
        
        # Get prescriptions with patient names
        cursor.execute("""
            SELECT 
                pr.prescription_id,
                pr.prescription_uuid,
                pr.patient_id,
                p.first_name || ' ' || p.last_name as patient_name,
                pr.drug_name,
                pr.quantity,
                pr.refills,
                pr.status,
                pr.prescribed_at
            FROM prescription_requests pr
            LEFT JOIN patients p ON pr.patient_id = p.patient_id
            ORDER BY pr.prescribed_at DESC
            LIMIT 100
        """)
        prescriptions = cursor.fetchall()
        cursor.close()
        
        return templates.TemplateResponse("prescriptions.html", {
            "request": request,
            "user": user,
            "prescriptions": prescriptions
        })


@app.get("/pharmacy/prescription/{prescription_id}", response_class=HTMLResponse)
def prescription_detail_page(request: Request, prescription_id: int, user_id: int = None):
    """Prescription detail and review page."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get user info
        if user_id:
            cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = cursor.fetchone()
            if not user:
                return RedirectResponse(url="/pharmacy/login")
        else:
            return RedirectResponse(url="/pharmacy/login")
        
        # Get prescription details
        cursor.execute("""
            SELECT 
                pr.*,
                p.first_name || ' ' || p.last_name as patient_name
            FROM prescription_requests pr
            LEFT JOIN patients p ON pr.patient_id = p.patient_id
            WHERE pr.prescription_id = %s
        """, (prescription_id,))
        prescription = cursor.fetchone()
        
        if not prescription:
            cursor.close()
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        # Get review results if exist
        cursor.execute("""
            SELECT 
                prv.*,
                u.full_name as reviewer_name
            FROM prescription_review prv
            LEFT JOIN users u ON prv.reviewed_by = u.user_id
            WHERE prv.prescription_id = %s
        """, (prescription_id,))
        review = cursor.fetchone()
        cursor.close()
        
        # Determine if prescription can be edited
        # Can edit if status is AI_APPROVED or AI_FLAGGED or AI_ERROR
        can_edit = prescription['status'] in ['AI_APPROVED', 'AI_FLAGGED', 'AI_ERROR']
        
        return templates.TemplateResponse("prescription_detail.html", {
            "request": request,
            "user": user,
            "prescription": prescription,
            "review": review,
            "can_edit": can_edit
        })


@app.post("/pharmacy/prescription/{prescription_id}/action")
async def take_action(
    request: Request,
    prescription_id: int,
    user_id: int = Form(...),
    action: str = Form(...),
    action_reason: str = Form(None)
):
    """Pharmacist takes action on prescription."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get prescription details
        cursor.execute(
            "SELECT * FROM prescription_requests WHERE prescription_id = %s",
            (prescription_id,)
        )
        prescription = cursor.fetchone()
        
        if not prescription:
            cursor.close()
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        # Update prescription_review with pharmacist action
        cursor.execute("""
            UPDATE prescription_review
            SET reviewed_by = %s,
                action = %s,
                action_reason = %s,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE prescription_id = %s
            RETURNING id
        """, (user_id, action, action_reason, prescription_id))
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            raise HTTPException(status_code=404, detail="Review record not found")
        
        # Update prescription status
        cursor.execute("""
            UPDATE prescription_requests
            SET status = %s
            WHERE prescription_id = %s
        """, (action, prescription_id))
        
        conn.commit()
        cursor.close()
        
        # Call doctor API to update status
        try:
            print(f"[DEBUG] Received action from form: '{action}' (type: {type(action).__name__})")
            
            # Map pharmacy action to doctor system status
            if action == 'APPROVED':
                api_action = "pharmacy_approved"
            elif action == 'DECLINED':
                api_action = "pharmacy_denied"
            else:
                api_action = action.lower()
            
            print(f"[DEBUG] Mapped to api_action: '{api_action}'")
            print(f"[DEBUG] Sending to doctor API - UUID: {prescription['prescription_uuid']}, Status: {api_action}, Notes: {action_reason}")

            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {
                    "prescription_uuid": prescription['prescription_uuid'],
                    "status": api_action,
                    "pharmacy_notes": action_reason
                }
                print(f"[DEBUG] Full payload: {payload}")
                
                doctor_response = await client.post(
                    f"{DOCTOR_API_URL}/api/prescription-status-update",
                    json=payload
                )
                
                if doctor_response.status_code == 200:
                    print(f"✓ Doctor system updated for prescription {prescription['prescription_uuid']}")
                else:
                    print(f"⚠️  Doctor API returned status {doctor_response.status_code}")
        except Exception as e:
            print(f"⚠️  Error calling doctor API: {e}")
        
        # Log pharmacist decision to blockchain
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                blockchain_response = await client.post(
                    "http://localhost:8001/pharmacy/pharmacist-decision",
                    json={
                        "prescription_uuid": prescription['prescription_uuid'],
                        "pharmacist_id": str(user_id),
                        "action": action,
                        "action_reason": action_reason or ""
                    }
                )
                if blockchain_response.status_code == 200:
                    print(f"✓ Pharmacist decision logged to blockchain for {prescription['prescription_uuid']}")
                else:
                    print(f"⚠️  Blockchain pharmacist logging failed: {blockchain_response.status_code}")
        except Exception as bc_err:
            print(f"⚠️  Blockchain pharmacist logging error: {bc_err}")
        
        # Redirect back to prescriptions list
        return RedirectResponse(
            url=f"/pharmacy/prescriptions?user_id={user_id}",
            status_code=303
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
