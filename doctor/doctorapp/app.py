from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
import httpx
import asyncio
import os
import json
import base64
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from typing import Optional, Tuple, Any, Dict, List

# Load environment variables
load_dotenv()

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'doctor_office'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('APP_PORT', '8003'))
PHARMACY_API_URL = os.getenv('PHARMACY_API_URL', 'http://localhost:8004')
QKD_SERVICE_URL = os.getenv('QKD_SERVICE_URL', 'http://localhost:8005')
ENABLE_QKD_ENCRYPTION = os.getenv('ENABLE_QKD_ENCRYPTION', 'true').lower() == 'true'
BLOCKCHAIN_URL = os.getenv('BLOCKCHAIN_URL', 'http://localhost:8001')

app = FastAPI(title="Doctor Office")

# Templates
templates = Jinja2Templates(directory="templates")

# Database functions
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

def db_execute_returning(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    result = None
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
    return result

# encrypt data using qkd key - AES encryption
def encrypt_with_qkd_key(data: Dict[str, Any], qkd_key_hex: str) -> str:
    
    # Convert hex key to bytes
    key_bytes = bytes.fromhex(qkd_key_hex)
    
    # Create Fernet cipher (uses AES-128 in CBC mode with HMAC)
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    cipher = Fernet(fernet_key)
    
    # Encrypt the JSON data
    plaintext = json.dumps(data).encode('utf-8')
    ciphertext = cipher.encrypt(plaintext)
    
    # Return as base64 string for JSON transmission
    return base64.b64encode(ciphertext).decode('utf-8')

# Get QKD key from service
async def get_qkd_key(user_id: Optional[int] = None, session_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    encryption_key = None
    qkd_session_id = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Initiate QKD session
            qkd_initiate_response = await client.post(f"{QKD_SERVICE_URL}/qkd/initiate",
                json={
                    "sender": f"doctor-office-{user_id}",
                    "receiver": "pharmacy-app",
                    "session_key": str(session_key)
                }
            )
            
            if qkd_initiate_response.status_code != 200:
                raise Exception(f"QKD initiation failed: {qkd_initiate_response.text}")
            
            qkd_initiate_data = qkd_initiate_response.json()
            qkd_session_id = qkd_initiate_data['session_id']
            
            # Complete BB84 key exchange and receive key directly
            # In real QKD,  key is derived from the quantum channel
            qkd_exchange_response = await client.post(f"{QKD_SERVICE_URL}/qkd/exchange",
                json={
                    "session_id": qkd_session_id,
                    "party": "sender" 
                }
            )
            
            if qkd_exchange_response.status_code != 200:
                raise Exception(f"QKD exchange failed: {qkd_exchange_response.text}")
            
            qkd_exchange_data = qkd_exchange_response.json()
            if not qkd_exchange_data['success']:
                raise Exception(f"QKD exchange unsuccessful: {qkd_exchange_data['message']}")
            
            encryption_key = qkd_exchange_data['key']
            
    except Exception as qkd_err:
        print(f"QKD encryption failed: {qkd_err}")
    
    return encryption_key, qkd_session_id

async def delete_qkd_session(qkd_session_id: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.delete(
                f"{QKD_SERVICE_URL}/qkd/destroy/{qkd_session_id}"
            )
              
    except Exception as destroy_err:
        print(f"QKD key destruction error: {destroy_err}")

# send transaction to blockchain
async def blockchain_transaction(endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{BLOCKCHAIN_URL}{endpoint}",json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Blockchain transaction failed: {response.text}")
                return None
    except Exception as e:
        print(f"Blockchain service unavailable: {e}")
        return None
    
# log doctor/pharamcy communication to a db a table (just for checking)
def log_communication(prescription_uuid: str, 
                     action_type: str, user_type: str, user_id: int, 
                     user_name: str, comments: str, previous_status: str, new_status: str) -> None:

    query = """
                INSERT INTO prescription_communications 
                (prescription_uuid, action_type, actor_type, 
                actor_id, actor_name, comments, previous_status, new_status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
    db_execute(query,   
            (prescription_uuid, action_type, user_type, 
             user_id, user_name, comments, previous_status, new_status))

############# PAGES ################

@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("login.html", {"request": request})

# Login page
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("login.html", {"request": request})

# Create prescription page
@app.get("/create-prescription", response_class=HTMLResponse)
def create_prescription_page(request: Request, user_id: Optional[int] = None) -> HTMLResponse:

    # check user
    if user_id is None:
        RedirectResponse(url="/login", status_code=303)
    
    user = db_query_one("SELECT * FROM users WHERE user_id = %s", (user_id,))

    if user is None:
        RedirectResponse(url="/login", status_code=303)

    # Get patients list sample 100 patients from db
    patients = db_query_all("SELECT * FROM patients ORDER BY last_name, first_name LIMIT 100")
        
    # Drug list (Sample drug list grouped by type to test)
    drugs = {
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
    
    return templates.TemplateResponse("create_prescription.html", {
        "request": request,
        "user": user,
        "patients": patients,
        "drugs": drugs
    })

# Prescriptions list page
@app.get("/prescriptions", response_class=HTMLResponse)
def prescriptions_page(request: Request, user_id: Optional[int] = None) -> HTMLResponse:
    # check user
    if user_id is None:
        RedirectResponse(url="/login", status_code=303)

    user = db_query_one("SELECT * FROM users WHERE user_id = %s", (user_id,))
    if user is None:
        RedirectResponse(url="/login", status_code=303)
    # Get prescriptions
    query = """
        SELECT 
            pr.id,
            pr.prescription_uuid,
            pr.patient_id,
            p.first_name || ' ' || p.last_name as patient_name,
            pr.drug_name,
            SPLIT_PART(pr.drug_name, ' ', 2) as dosage,
            pr.quantity,
            pr.refills,
            pr.status,
            pr.pharmacy_notes,
            pr.created_at,
            u.full_name as doctor_name
        FROM prescription_requests pr
        JOIN patients p ON pr.patient_id = p.patient_id
        JOIN users u ON pr.prescriber_id = u.user_id
        ORDER BY pr.created_at DESC
    """
    
    prescriptions = db_query_all(query)

    return templates.TemplateResponse("prescriptions.html", {
        "request": request,
        "user": user,
        "prescriptions": prescriptions
    })

# Prescription review detail page
@app.get("/prescription/{prescription_id}/review", response_class=HTMLResponse)
def prescription_review_page(request: Request, prescription_id: int, user_id: Optional[int] = None) -> HTMLResponse:

    # check user
    if user_id is None:
        RedirectResponse(url="/login", status_code=303)
    
    user = db_query_one("SELECT * FROM users WHERE user_id = %s", (user_id,))
    if user is None:
        RedirectResponse(url="/login", status_code=303)
    
    # Get prescription details
    query = """
        SELECT 
            pr.id,
            pr.prescription_uuid,
            pr.patient_id,
            p.first_name || ' ' || p.last_name as patient_name,
            pr.drug_name,
            pr.quantity,
            pr.refills,
            pr.status,
            pr.pharmacy_notes,
            pr.created_at,
            pr.updated_at
        FROM prescription_requests pr
        JOIN patients p ON pr.patient_id = p.patient_id
        WHERE pr.id = %s AND pr.prescriber_id = %s
    """
    prescription = db_query_one(query, (prescription_id, user_id))
    
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    # Get communication history
    communications = db_query_all("""
        SELECT * FROM prescription_communications
        WHERE prescription_uuid = %s
        ORDER BY created_at ASC
    """, (prescription['prescription_uuid'],))
    
    return templates.TemplateResponse("prescription_review.html", {
        "request": request,
        "user": user,
        "prescription": prescription,
        "communications": communications
    })

############ API ENDPOINTS ################
# Login endpoint
@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)) -> HTMLResponse:

    user = db_query_one("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    
    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid credentials"
        })
    
    # Redirect to prescriptions list with user info in session (simple approach)
    response = RedirectResponse(url=f"/prescriptions?user_id={user['user_id']}", status_code=303)
    return response

# Create prescription API
@app.post("/create-prescription")
async def create_prescription(
    user_id: int = Form(...), 
    patient_id: int = Form(...), 
    drug_name: str = Form(...), 
    quantity: int = Form(...), 
    refills: int = Form(...)
) -> RedirectResponse:
    
    # Insert prescription into DB
    query = """
        INSERT INTO prescription_requests 
        (patient_id, prescriber_id, drug_name, quantity, refills, status)
        VALUES (%s, %s, %s, %s, %s, 'sent_to_pharmacy')
        RETURNING prescription_uuid
    """
    
    result = db_execute_returning(query, (
        patient_id, user_id, drug_name, quantity, refills
    ))

    prescription_uuid = result['prescription_uuid']

    # Log to blockchain FIRST (before sending to pharmacy)
    await blockchain_transaction("/pharmacy/prescription-created", {
        "prescription_uuid": prescription_uuid,
        "doctor_id": str(user_id)
    })
    
    # Prescription data in JSON to send to Pharamcy
    prescription_data_to_send = {
        "patient_id": patient_id,
        "prescription_uuid": prescription_uuid,
        "drug_name": drug_name,
        "quantity": quantity,
        "refills": refills
    }
    
    # if QKD is enabled, encrypt prescription json above before sending to pharmacy
    qkd_session_id = None
       
    if ENABLE_QKD_ENCRYPTION:
        encryption_key, qkd_session_id = await get_qkd_key()

        if encryption_key:
            # Encrypt prescription data with quantum-safe key
            encrypted_payload = encrypt_with_qkd_key(prescription_data_to_send, encryption_key)
            
            # delete key from memory
            del encryption_key
            
            # Update payload for pharmacy
            prescription_data_to_send = {
                "encrypted": True,
                "qkd_session_id": qkd_session_id,
                "ciphertext": encrypted_payload,
                "prescription_uuid": prescription_uuid  # Include for routing
            }
                
    # Send to pharmacy API
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            pharmacy_response = await client.post(
                f"{PHARMACY_API_URL}/prescription",
                json=prescription_data_to_send
            )        
    except httpx.ConnectError:
        print(f"Pharmacy API not reachable at {PHARMACY_API_URL}")
    except Exception as e:
        print(f"Error calling pharmacy API: {e}")
    
    # Delete QKD session after key use by both parties
    if qkd_session_id:
        await delete_qkd_session(qkd_session_id)
    
    # Redirect back to prescriptions list
    return RedirectResponse(url=f"/prescriptions?user_id={user_id}", status_code=303)


# API endpoint to receive prescription status update from pharmacy
@app.post("/api/prescription-status-update")
async def update_prescription_status(request: Request) -> JSONResponse:

    update = await request.json()
    prescription_uuid = update.get('prescription_uuid')
    status = update.get('status')
    pharmacy_notes = update.get('pharmacy_notes')
        
    # Update prescription status
    query = """
        UPDATE prescription_requests
        SET status = %s, 
            pharmacy_notes = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE prescription_uuid = %s
        RETURNING id, prescription_uuid, status
    """
    
    # Get current status before update for communication logging
    prescription = db_query_one("SELECT status FROM prescription_requests WHERE prescription_uuid = %s", (prescription_uuid,))
    previous_status = prescription['status'] if prescription else 'unknown'
    
    result = db_execute_returning(query, (status, pharmacy_notes, prescription_uuid))

    if not result:
        print(f"[ERROR] Prescription {prescription_uuid} not found in database")
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"Prescription {prescription_uuid} not found"}
        )
       
    # Log communication from pharmacy 
    if status == 'pending_review' and pharmacy_notes:
        log_communication(
            prescription_uuid,
            'PHARMACIST_REQUEST_REVIEW', 'PHARMACIST', 0,
            "Pharmacist", pharmacy_notes,
            previous_status, status
        )
    
    result_dict = {
        "id": result['id'],
        "prescription_uuid": result['prescription_uuid'],
        "status": result['status']
    }
            
    return JSONResponse(content={
        "success": True,
        "message": "Prescription status updated successfully",
        "prescription_uuid": result_dict['prescription_uuid'],
        "status": result_dict['status']
    })
  
# API to get prescriptions under review
@app.get("/api/prescriptions-under-review")
def get_prescriptions_under_review(user_id: int) -> JSONResponse:

    query = """
        SELECT 
            pr.id,
            pr.prescription_uuid,
            pr.patient_id,
            p.first_name || ' ' || p.last_name as patient_name,
            pr.drug_name,
            pr.quantity,
            pr.refills,
            pr.status,
            pr.pharmacy_notes,
            pr.created_at,
            pr.updated_at
        FROM prescription_requests pr
        JOIN patients p ON pr.patient_id = p.patient_id
        WHERE pr.prescriber_id = %s 
        AND pr.status IN ('pending_review', 'under_review')
        ORDER BY pr.updated_at DESC
    """
    
    prescriptions = db_query_all(query, (user_id,))
    # Convert to serializable format
    result = []
    for p in prescriptions:
        p_dict = dict(p)
        # Convert datetime objects to strings
        if p_dict.get('created_at'):
            p_dict['created_at'] = p_dict['created_at'].isoformat()
        if p_dict.get('updated_at'):
            p_dict['updated_at'] = p_dict['updated_at'].isoformat()
        result.append(p_dict)
    
    return JSONResponse(content={
        "success": True,
        "prescriptions": result
    })
 
# Respond to review API
@app.post("/prescription/{prescription_id}/respond")
async def respond_to_review(
    request: Request,
    prescription_id: int,
    user_id: int = Form(...),
    response_action: str = Form(...),
    response_comments: str = Form(...)
) -> RedirectResponse:

    # Get prescription details
    prescription = db_query_one(
        "SELECT * FROM prescription_requests WHERE id = %s AND prescriber_id = %s",
        (prescription_id, user_id)
    )
    
    if not prescription:
        raise HTTPException(status_code=404, detail="Prescription not found")
    
    # Get doctor details
    doctor = db_query_one("SELECT full_name FROM users WHERE user_id = %s", (user_id,))
    doctor_name = doctor['full_name'] if doctor else f"Doctor {user_id}"
    
    previous_status = prescription['status']

    if response_action == 'cancel':
        new_status = 'cancelled'
        db_log = 'DOCTOR_CANCEL'
        blockchain_endpoint = 'cancel-prescription'
        blockchain_payload = {
            "prescription_uuid": prescription['prescription_uuid'],
            "doctor_id": str(user_id),
            "cancellation_reason": response_comments or "Cancelled by doctor"
        }
        pharmacy_notify_payload = {
            "prescription_uuid": prescription['prescription_uuid'], 
            "status": "CANCELLED",
            "pharmacy_notes": response_comments or "Cancelled by doctor"
        }


    else:
        new_status = 'under_review'
        db_log = 'DOCTOR_RESPONSE'
        blockchain_endpoint = 'respond-to-review'
        blockchain_payload = {
            "prescription_uuid": prescription['prescription_uuid'],
            "doctor_id": str(user_id),
            "response_comments": response_comments or ""
        }
        pharmacy_notify_payload = {
            "prescription_uuid": prescription['prescription_uuid'],
            "status": "UNDER_REVIEW",
            "pharmacy_notes": response_comments or ""
        }

    # update prescription status in DB
    db_execute("""
        UPDATE prescription_requests
        SET status = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
    """, (new_status, prescription_id))
    
    # Log prescription communications to db
    log_communication(
            prescription['prescription_uuid'],
            db_log, 'DOCTOR', user_id,
            doctor_name, response_comments or "", 
            previous_status, new_status
        )
    
    #log to blockchain
    await blockchain_transaction(f"/doctor/{blockchain_endpoint}", blockchain_payload)
    
    #notify pharmacy
    try:
        print(f"Notifying pharmacy of doctor response for prescription {prescription['prescription_uuid']}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{PHARMACY_API_URL}/api/prescription-status-update", json=pharmacy_notify_payload)
                
    except Exception as e:
        print(f"Error notifying pharmacy: {e}")
    
    return RedirectResponse(url=f"/prescriptions?user_id={user_id}", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
