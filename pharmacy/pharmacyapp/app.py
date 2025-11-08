"""
Pharmacy App
"""

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
import torch.nn as nn
import numpy as np
import os
from dotenv import load_dotenv
from feature_extractor import FeatureExtractor
import httpx
import asyncio
import json
import base64
from cryptography.fernet import Fernet
from typing import Optional, Tuple, Any, Dict, List

# Load environment variables
load_dotenv()

# DB Configuration
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

# QKD service configuration
QKD_SERVICE_URL = os.getenv('QKD_SERVICE_URL', 'http://localhost:8005')
# Initialize FastAPI app
app = FastAPI()

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

def get_db() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def db_query_one(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()
        cursor.close()
        return result

def db_query_all(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return results  

def db_execute(query: str, params: tuple = ()) -> Any:
    result = None
    with get_db() as conn:
        cursor = conn.cursor()
        result = cursor.execute(query, params)
        conn.commit()
        cursor.close()
    return result

async def get_qkd_key(session_id: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{QKD_SERVICE_URL}/qkd/exchange",
                json={
                    "session_id": session_id,
                    "party": "receiver"
                }
            )
            if response.status_code == 200:
                qkd_data = response.json()
                if qkd_data['success']:
                    return qkd_data['key']
                else:
                    raise Exception(f"QKD key retrieval unsuccessful: {qkd_data['message']}")
            else:
                raise Exception(f"QKD key retrieval failed: {response.text}")
    except Exception as e:
        raise Exception(f"QKD service unavailable: {e}")

# descrypt - return disctionary (aks json)
def decrypt_with_qkd_key(ciphertext_b64: str, qkd_key_hex: str) -> Dict[str, Any]:

    # Convert hex key to bytes
    key_bytes = bytes.fromhex(qkd_key_hex)
    
    # Create Fernet cipher (requires base64-encoded key)
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    cipher = Fernet(fernet_key)
    
    # Decode base64 ciphertext
    ciphertext = base64.b64decode(ciphertext_b64.encode('utf-8'))
    
    # Decrypt
    plaintext = cipher.decrypt(ciphertext)
    
    # Parse JSON
    data = json.loads(plaintext.decode('utf-8'))
    
    return data

# log into prescription_communications table
def log_communication(prescription_uuid: str, action_type: str, user_type: str, user_id: int, actor_name: str, comments: str, previous_status: str, new_status: str) -> None:

    prescription = db_query_one("SELECT * FROM prescription_requests WHERE prescription_uuid = %s", (prescription_uuid,))
    if not prescription:
        print(f"Prescription {prescription_uuid} not found for logging communication")
        return
    
    # pharamacy db id of presecription
    prescription_id = prescription['prescription_id']

    # insert into prescription_communications table
    db_execute("""
        INSERT INTO prescription_communications
            (prescription_uuid, prescription_id, action_type, actor_type, 
            actor_id, actor_name, comments, previous_status, new_status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """, 
        (prescription_uuid, prescription_id, action_type, user_type,
        user_id, actor_name, comments, previous_status, new_status))

# send transaction to blockchain
async def blockchain_transaction(endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{BLOCKCHAIN_SERVICE_URL}{endpoint}",json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Blockchain transaction failed: {response.text}")
                return None
    except Exception as e:
        print(f"Blockchain service unavailable: {e}")
        return None

# Load and run models

MODEL_DIR = '../../ai-layer/model'
ELIGIBILITY_MODEL_PATH = os.path.join(MODEL_DIR, 'results/50000_v3/dnn_eligibility_model.pth')
OUD_MODEL_PATH = os.path.join(MODEL_DIR, 'results/10000_v3/dnn_oud_risk_model.pth')

# Feature lists will be loaded from model checkpoints during model loading
# DO NOT hardcode these - they MUST match exactly what the models were trained with
ELIGIBILITY_FEATURES = None  # Will be set by load_models()
OUD_FEATURES = None  # Will be set by load_models()

class DeepNet(nn.Module):
    """Deep Neural Network with BatchNorm - matches training architecture."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32, 16], dropout_rate: float = 0.3):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Load eligibility and OUD Risk Models from trained model checkpoints
def load_models() -> Tuple[nn.Module, Any, nn.Module, Any]:
    global ELIGIBILITY_FEATURES, OUD_FEATURES
    
    # Load Eligibility model checkpoint
    eligibility_checkpoint = torch.load(ELIGIBILITY_MODEL_PATH, map_location='cpu', weights_only=False)
    ELIGIBILITY_FEATURES = eligibility_checkpoint['feature_cols'] 
    eligibility_input_dim = eligibility_checkpoint.get('input_dim', len(ELIGIBILITY_FEATURES))
    
    eligibility_model = DeepNet(input_dim=eligibility_input_dim)
    eligibility_model.load_state_dict(eligibility_checkpoint['model_state_dict'])
    eligibility_model.eval()
    eligibility_scaler = eligibility_checkpoint.get('scaler', None)
    
    # Load OUD Risk model checkpoint
    oud_checkpoint = torch.load(OUD_MODEL_PATH, map_location='cpu', weights_only=False)
    OUD_FEATURES = oud_checkpoint['feature_cols']  # Get correct feature order from checkpoint
    oud_input_dim = oud_checkpoint.get('input_dim', len(OUD_FEATURES))
    
    oud_model = DeepNet(input_dim=oud_input_dim)
    oud_model.load_state_dict(oud_checkpoint['model_state_dict'])
    oud_model.eval()
    oud_scaler = oud_checkpoint.get('scaler', None)
    
    return eligibility_model, eligibility_scaler, oud_model, oud_scaler


# Global model instances
eligibility_model, eligibility_scaler, oud_model, oud_scaler = load_models()

# Check prescription with AI in background
async def process_prescription_ai_check(prescription_id: int, patient_id: int, drug_name: str, prescription_uuid: str) -> None:

    try:
        
        # Check if drug is an opioid
        opioid_keywords = [
            'fentanyl', 'morphine', 'oxycodone', 'hydrocodone', 'hydromorphone',
            'oxymorphone', 'codeine', 'tramadol', 'methadone', 'buprenorphine',
            'meperidine', 'tapentadol', 'opioid', 'narcotic'
        ]
        
        drug_name_lower = drug_name.lower()
        is_opioid = any(keyword in drug_name_lower for keyword in opioid_keywords)

        if not is_opioid:
            # Non-opioid prescriptions are automatically approved
            db_execute("""
                INSERT INTO prescription_review (
                            prescription_id, eligibility_score, eligibility_prediction,
                            oud_risk_score, oud_risk_prediction,
                            flagged, flag_reason, recommendation
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        prescription_id, 1.0, 1,
                        0.0, 0,
                        False, "Non-opioid prescription", "AI APPROVED: Non-opioid prescription"
                    ))
            #update prescription status
            db_execute("""
                UPDATE prescription_requests
                SET status = 'AI_APPROVED'
                WHERE prescription_id = %s
            """, (prescription_id,))
            
            await blockchain_transaction(f"/pharmacy/ai-review", {
                "prescription_uuid": prescription_uuid,
                "flagged": False,
                "eligibility_score": 100,
                "oud_risk_score": 0,
                "flag_reason": "Non-opioid prescription",
                "recommendation": "AI APPROVED: Non-opioid prescription"
            })
            return
        
        # Initialize feature extractor
        extractor = FeatureExtractor(DB_CONFIG)
        
        # Calculate features 
        eligibility_features_dict = extractor.extract_eligibility_features(str(patient_id))
        oud_features_dict = extractor.extract_oud_features(str(patient_id))
        
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
        
        db_execute("""
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
        #update prescription status
        db_execute("""
            UPDATE prescription_requests
            SET status = %s
            WHERE prescription_id = %s
        """, (new_status, prescription_id))


        await blockchain_transaction(f"/pharmacy/ai-review", {
            "prescription_uuid": prescription_uuid,
            "flagged": bool(flagged),
            "eligibility_score": int(eligibility_score * 100),
            "oud_risk_score": int(oud_score * 100),
            "flag_reason": flag_reason,
            "recommendation": recommendation
        })
            
    except Exception as e:
        print(f"ERROR in AI check for prescription_id={prescription_id}: {e}")
        
        # Update status to AI_ERROR
        db_execute("""
            UPDATE prescription_requests
            SET status = 'AI_ERROR'
            WHERE prescription_id = %s
            """, (prescription_id,))

# fetch patient features from database
def get_patient_features(conn: psycopg2.extensions.connection, patient_id: int) -> Dict[str, Any]:

    patient = db_query_one("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))    
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    return dict(patient)

# prepare features for model input
def prepare_features(patient_data: Dict[str, Any], feature_list: List[str]) -> np.ndarray:
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

# run inteference for the model
def run_inference(model: nn.Module, features: np.ndarray) -> Tuple[float, int]:
    """Run model inference and return probability."""
    with torch.no_grad():
        x_tensor = torch.tensor(features, dtype=torch.float32)
        logits = model(x_tensor).item()
        # Apply sigmoid to convert logits to probability
        prob = 1 / (1 + np.exp(-logits))
        prediction = 1 if prob >= 0.5 else 0
    
    return prob, prediction

############### APIs ###############

# Login API
@app.post("/pharmacy/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)) -> HTMLResponse:

    # Simple authentication check
    user = db_query_one("SELECT user_id, email, password, full_name, role FROM users WHERE email = %s AND is_active = TRUE", (email,))
    
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


# New prescription API (from doctors app)
@app.post("/prescription")
async def receive_prescription(request: Request) -> Dict[str, Any]:

    request_data = await request.json()
    
    # Handle QKD decryption if needed
    if request_data.get('encrypted'):

        # get qkd key
        qkd_key = await get_qkd_key(request_data['qkd_session_id'])
        if not qkd_key:
            raise Exception("Failed to retrieve QKD key")
                        
        # decrypt prescription data
        decrypted_data = decrypt_with_qkd_key(request_data['ciphertext'], qkd_key)
        
        # Wipe key from local memory immediately
        del qkd_key
        
        # Update request_data with decrypted data
        request_data['patient_id'] = decrypted_data['patient_id']
        request_data['prescription_uuid'] = decrypted_data['prescription_uuid']
        request_data['drug_name'] = decrypted_data['drug_name']
        request_data['quantity'] = decrypted_data.get('quantity')
        request_data['refills'] = decrypted_data.get('refills')     
    else:
        print(f"Received unencrypted prescription: uuid={request_data['prescription_uuid']}, patient={request_data['patient_id']}, drug={request_data['drug_name']}")
  
    # Insert prescription with RECEIVED status into prescription_requests table
    db_execute("""
                INSERT INTO prescription_requests (
                    patient_id, drug_name, prescription_uuid, 
                    quantity, refills, status
                ) VALUES (%s, %s, %s, %s, %s, 'RECEIVED')
                RETURNING prescription_id
                """, 
                (request_data['patient_id'], request_data['drug_name'], request_data['prescription_uuid'], request_data['quantity'], request_data['refills']))

    # get the inserted prescription_id

    prescription = db_query_one("SELECT prescription_id FROM prescription_requests WHERE prescription_uuid = %s", (request_data['prescription_uuid'],))
    prescription_id = prescription['prescription_id']
    
    # Kick off AI check in background (fire and forget)
    asyncio.create_task(
        process_prescription_ai_check(
            prescription_id=prescription_id,
            patient_id=request_data['patient_id'],
            drug_name=request_data['drug_name'],
            prescription_uuid=request_data['prescription_uuid']
        )
    )
    
    # Return immediate success response
    return {
        "status": "success",
        "prescription_uuid": request_data['prescription_uuid'],
        "message": "Prescription received and queued for review"
    }
            
# API for doctors reponse to review request
@app.post("/api/prescription-status-update")
async def update_prescription_status_from_doctor(request: Request) -> Dict[str, Any]:

    request_data = await request.json()

    prescription = db_query_one("SELECT * FROM prescription_requests WHERE prescription_uuid = %s", (request_data['prescription_uuid'],))

    if not prescription:
        return {
            "success": False,
            "message": f"Prescription {request_data['prescription_uuid']} not found in pharmacy database"
        }
    
    previous_status = prescription['status']
    
    # Update prescription status in pharmacy database
    db_execute("""
                        UPDATE prescription_requests
                        SET status = %s
                        WHERE prescription_uuid = %s
                        """, 
                        (request_data['status'], request_data['prescription_uuid'])) 
    
    # Log to communications table
    action_type = 'DOCTOR_CANCEL' if request_data['status'] == 'CANCELLED' else 'DOCTOR_RESPONSE'
    log_communication(
        request_data['prescription_uuid'],
        action_type, 'DOCTOR', 0, 
        "Doctor", request_data.get('pharmacy_notes', ''), 
        previous_status, request_data['status']
    )
    
    return {
        "success": True,
        "message": "Prescription status updated in pharmacy database",
        "prescription_uuid": request_data['prescription_uuid'],
        "status": request_data['status']
    }



# api for pharmacist to take action on prescription
@app.post("/pharmacy/prescription/{prescription_id}/action")
async def take_action(
    request: Request,
    prescription_id: int,
    user_id: int = Form(...),
    action: str = Form(...),
    action_reason: str = Form(None)
) -> RedirectResponse:
    
    prescription = db_query_one("SELECT * FROM prescription_requests WHERE prescription_id = %s", (prescription_id,))
    
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    # Get pharmacist details

    pharmacist = db_query_one("SELECT user_id, full_name FROM users WHERE user_id = %s", (user_id,))

    pharmacist_name = pharmacist['full_name'] if pharmacist else f"Pharmacist {user_id}"
    
    previous_status = prescription['status']

    match action:
        case 'APPROVED':
            new_status = 'APPROVED'
            doctor_app_status = 'pharmacy_approved'
            blockchain_svc_endpoint = "/pharmacy/pharmacist-decision"
            blockchain_request_payload = {
                "prescription_uuid": prescription['prescription_uuid'],
                "pharmacist_id": str(user_id),
                "action": action,
                "action_reason": action_reason or ""
            }

        case 'DECLINED':
            new_status = 'DECLINED'
            doctor_app_status = 'pharmacy_denied'
            blockchain_svc_endpoint = "/pharmacy/pharmacist-decision"
            blockchain_request_payload = {
                "prescription_uuid": prescription['prescription_uuid'],
                "pharmacist_id": str(user_id),
                "action": action,
                "action_reason": action_reason or ""
            }

        case 'REQUEST_REVIEW':
            new_status = 'PENDING_REVIEW'
            doctor_app_status = 'pending_review'
            blockchain_svc_endpoint = "/pharmacy/request-review"
            blockchain_request_payload = {
                "prescription_uuid": prescription['prescription_uuid'],
                "pharmacist_id": str(user_id),
                "pharmacist_name": pharmacist_name,
                "review_comments": action_reason or ""
            }

    # For approved or denied prescriptions, update in review table
    if action in ['APPROVED', 'DECLINED']:
        # Update prescription_review table
        db_execute("""
            UPDATE prescription_review
            SET reviewed_by = %s,
                action = %s,
                action_reason = %s,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE prescription_id = %s
        """, 
        (user_id, action, action_reason, prescription_id))

    # update status in prescription_requests table
    db_execute("UPDATE prescription_requests SET status = %s WHERE prescription_id = %s", (new_status, prescription_id))

    # insert into prescription communications table
    log_communication(prescription['prescription_uuid'], 'PHARMACIST_ACTION', 'PHARMACIST', user_id, pharmacist_name, action_reason or "", previous_status, new_status)

    # send to blockchain service
    await blockchain_transaction(blockchain_svc_endpoint, blockchain_request_payload)

    # update doctors system via API
    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {
            "prescription_uuid": prescription['prescription_uuid'],
            "status": doctor_app_status,
            "pharmacy_notes": action_reason or ""
        }
        try:
            doctor_response = await client.post(
                f"{DOCTOR_API_URL}/api/prescription-status-update",
                json=payload
            )
            if doctor_response.status_code == 200:
                print(f"Doctor system updated for prescription {prescription['prescription_uuid']}")
            else:
                print(f"Doctor API returned status {doctor_response.status_code}")
        except Exception as e:
            print(f"Error calling doctor API: {e}")
    
    return RedirectResponse(url=f"/pharmacy/prescriptions?user_id={user_id}",status_code=303)

############### PAGES ###############

# root page redirects to login
@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> RedirectResponse:
    """Redirect to login page."""
    return RedirectResponse(url="/pharmacy/login")


# login page
@app.get("/pharmacy/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("login.html", {"request": request})

# prescriptions list page 
@app.get("/pharmacy/prescriptions", response_class=HTMLResponse)
def prescriptions_page(request: Request, user_id: Optional[int] = None) -> HTMLResponse:
    
    # check if user logged in
    if not user_id:
        return RedirectResponse(url="/pharmacy/login")
    
    user = db_query_one("SELECT * FROM users WHERE user_id = %s", (user_id,))
    if not user:
        return RedirectResponse(url="/pharmacy/login")

    prescriptions = db_query_all("""
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

    return templates.TemplateResponse("prescriptions.html", {
        "request": request,
        "user": user,
        "prescriptions": prescriptions
    })

# prescription detail and review page
@app.get("/pharmacy/prescription/{prescription_id}", response_class=HTMLResponse)
def prescription_detail_page(request: Request, prescription_id: int, user_id: Optional[int] = None) -> HTMLResponse:

    # check if user logged in
    if not user_id:
        return RedirectResponse(url="/pharmacy/login")
    
    user = db_query_one("SELECT * FROM users WHERE user_id = %s", (user_id,))

    if not user:
        return RedirectResponse(url="/pharmacy/login")

    # Get prescription details
    prescription = db_query_one("""
                                SELECT 
                                    pr.*,
                                    p.first_name || ' ' || p.last_name as patient_name
                                FROM prescription_requests pr
                                LEFT JOIN patients p ON pr.patient_id = p.patient_id
                                WHERE pr.prescription_id = %s
                                """,
                                (prescription_id,))
                                       
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
     
    # Get review results if exist
    review = db_query_one("""
                        SELECT 
                            prv.*,
                            u.full_name as reviewer_name
                        FROM prescription_review prv
                        LEFT JOIN users u ON prv.reviewed_by = u.user_id
                        WHERE prv.prescription_id = %s
                        """, 
                        (prescription_id,))
       
    # Get communication history

    communications = db_query_all("""
                                    SELECT 
                                        communication_id,
                                        action_type,
                                        actor_type,
                                        actor_name,
                                        comments,
                                        previous_status,
                                        new_status,
                                        created_at
                                    FROM prescription_communications
                                    WHERE prescription_uuid = %s
                                    ORDER BY created_at ASC
                                """, 
                                (prescription['prescription_uuid'],))
    
    # Determine if prescription can be edited
    # Can edit if status is AI_APPROVED, AI_FLAGGED, AI_ERROR, or UNDER_REVIEW
    can_edit = prescription['status'] in ['AI_APPROVED', 'AI_FLAGGED', 'AI_ERROR', 'UNDER_REVIEW']
    print(f"Can edit prescription {prescription_id}: {can_edit} (status={prescription['status']})")
    
    return templates.TemplateResponse("prescription_detail.html", {
        "request": request,
        "user": user,
        "prescription": prescription,
        "review": review,
        "communications": communications,
        "can_edit": can_edit
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
