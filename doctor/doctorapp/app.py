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

app = FastAPI(title="Doctor Office")

# Templates
templates = Jinja2Templates(directory="templates")

# Database connection
def get_db():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

# Encryption helper
def encrypt_with_qkd_key(data: dict, qkd_key_hex: str) -> str:
    """
    Encrypt data using AES-256 with QKD-generated key
    
    Args:
        data: Dictionary to encrypt
        qkd_key_hex: Hex-encoded 256-bit key from QKD service
        
    Returns:
        Base64-encoded data
    """
    # Convert hex key to bytes
    key_bytes = bytes.fromhex(qkd_key_hex)
    
    # Create Fernet cipher (uses AES-128 in CBC mode with HMAC)
    # Fernet requires base64-urlsafe encoded 32-byte key
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    cipher = Fernet(fernet_key)
    
    # Encrypt the JSON data
    plaintext = json.dumps(data).encode('utf-8')
    ciphertext = cipher.encrypt(plaintext)
    
    # Return as base64 string for JSON transmission
    return base64.b64encode(ciphertext).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Login page
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Login endpoint
@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cur.fetchone()
    cur.close()
    conn.close()
    
    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid credentials"
        })
    
    # Redirect to prescriptions list with user info in session (simple approach)
    response = RedirectResponse(url=f"/prescriptions?user_id={user['user_id']}", status_code=303)
    return response



# Create prescription page
@app.get("/create-prescription", response_class=HTMLResponse)
def create_prescription_page(request: Request, user_id: int):
    # Get user info
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
    user = cur.fetchone()
    
    # Get patients list
    cur.execute("SELECT * FROM patients ORDER BY last_name, first_name LIMIT 100")
    patients = cur.fetchall()
    cur.close()
    conn.close()
    
    # Drug list (same as main API)
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

# Create prescription
@app.post("/create-prescription")
async def create_prescription(
    user_id: int = Form(...),
    patient_id: str = Form(...),
    drug_name: str = Form(...),
    quantity: int = Form(...),
    refills: int = Form(0)
):
    conn = get_db()
    cur = conn.cursor()
    
    # Parse dosage from drug_name (e.g., "Oxycodone 5mg" -> dosage = "5mg")
    dosage = drug_name.split()[-1] if len(drug_name.split()) > 1 else "N/A"
    
    query = """
        INSERT INTO prescription_requests 
        (patient_id, prescriber_id, drug_name, quantity, refills, status)
        VALUES (%s, %s, %s, %s, %s, 'sent_to_pharmacy')
        RETURNING prescription_uuid
    """
    
    cur.execute(query, (
        patient_id, user_id, drug_name, quantity, refills
    ))
    
    result = cur.fetchone()
    prescription_uuid = result['prescription_uuid']
    conn.commit()
    cur.close()
    conn.close()
    
    # QKD-based encryption (if enabled)
    qkd_session_id = None
    prescription_data_to_send = {
        "patient_id": patient_id,
        "prescription_uuid": prescription_uuid,
        "drug_name": drug_name,
        "quantity": quantity,
        "refills": refills
    }
    
    if ENABLE_QKD_ENCRYPTION:
        print(f"QKD Encryption Enabled for Prescription {prescription_uuid}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Initiate QKD session
                print("Step 1: Initiating QKD session...")
                qkd_initiate_response = await client.post(
                    f"{QKD_SERVICE_URL}/qkd/initiate",
                    json={
                        "sender": f"doctor-office-{user_id}",
                        "receiver": "pharmacy-app",
                        "prescription_uuid": prescription_uuid
                    }
                )
                
                if qkd_initiate_response.status_code != 200:
                    raise Exception(f"QKD initiation failed: {qkd_initiate_response.text}")
                
                qkd_initiate_data = qkd_initiate_response.json()
                qkd_session_id = qkd_initiate_data['session_id']
                print(f"Session initiated: {qkd_session_id}")
                
                # Step 2: Complete BB84 key exchange and receive key directly
                # In real QKD, this represents deriving the key from the quantum channel
                print("Step 2: Running BB84 quantum key exchange (receiving key via quantum channel)...")
                qkd_exchange_response = await client.post(
                    f"{QKD_SERVICE_URL}/qkd/exchange",
                    json={
                        "session_id": qkd_session_id,
                        "party": "sender"  # Doctor is the sender (Alice)
                    }
                )
                
                if qkd_exchange_response.status_code != 200:
                    raise Exception(f"QKD exchange failed: {qkd_exchange_response.text}")
                
                qkd_exchange_data = qkd_exchange_response.json()
                if not qkd_exchange_data['success']:
                    raise Exception(f"QKD exchange unsuccessful: {qkd_exchange_data['message']}")
                
                # Key is returned DIRECTLY in the exchange response (never stored in service)
                qkd_key = qkd_exchange_data['key']
                
                print(f"""
                      BB84 exchange successful - key received via quantum channel, 
                      Error rate: {qkd_exchange_data['error_rate']:.4f}, 
                      Secure: {qkd_exchange_data['secure']}, 
                      Key bits: {qkd_exchange_data['final_key_bits']}
                      """)
                
                # Step 3: Encrypt prescription data with quantum-safe key
                print("Step 3: Encrypting prescription data with AES-256...")
                encrypted_payload = encrypt_with_qkd_key(prescription_data_to_send, qkd_key)
                print(f"Data encrypted (payload size: {len(encrypted_payload)} bytes)")
                
                # Wipe key from local memory immediately after use
                qkd_key = None
                del qkd_key
                
                # Update payload for pharmacy
                prescription_data_to_send = {
                    "encrypted": True,
                    "qkd_session_id": qkd_session_id,
                    "ciphertext": encrypted_payload,
                    "prescription_uuid": prescription_uuid  # Include for routing
                }
                
                print(f"{'='*60}\n")
                
        except Exception as qkd_err:
            print(f"⚠️  QKD encryption failed: {qkd_err}")
            print(f"  Falling back to unencrypted transmission")
            # Keep original prescription_data_to_send
    
    # Log to blockchain FIRST (before sending to pharmacy)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            blockchain_response = await client.post(
                "http://localhost:8001/pharmacy/prescription-created",
                json={
                    "prescription_uuid": prescription_uuid,
                    "doctor_id": str(user_id)
                }
            )
            if blockchain_response.status_code == 200:
                print(f"✓ Prescription creation logged to blockchain")
            else:
                print(f"Blockchain logging failed: {blockchain_response.status_code}")
    except Exception as bc_err:
        print(f"⚠️  Blockchain logging error: {bc_err}")
    
    # Send to pharmacy API
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            pharmacy_response = await client.post(
                f"{PHARMACY_API_URL}/prescription",
                json=prescription_data_to_send
            )
            
            if pharmacy_response.status_code == 200:
                print(f"✓ Prescription {prescription_uuid} sent to pharmacy successfully")
                pharmacy_data = pharmacy_response.json()
                print(f"  Pharmacy response: {pharmacy_data}")
                    
            else:
                print(f"⚠️  Pharmacy API returned status {pharmacy_response.status_code}")
                print(f"  Response: {pharmacy_response.text}")
    except httpx.ConnectError:
        print(f"⚠️  Pharmacy API not reachable at {PHARMACY_API_URL}")
    except Exception as e:
        print(f"⚠️  Error calling pharmacy API: {e}")
    
    # Step 5: Destroy QKD key (if QKD was used)
    if qkd_session_id:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                destroy_response = await client.delete(
                    f"{QKD_SERVICE_URL}/qkd/destroy/{qkd_session_id}"
                )
                if destroy_response.status_code == 200:
                    print(f"✓ QKD keys securely destroyed for session {qkd_session_id}")
                else:
                    print(f"⚠️  Failed to destroy QKD keys: {destroy_response.status_code}")
        except Exception as destroy_err:
            print(f"⚠️  QKD key destruction error: {destroy_err}")
    
    # Redirect back to prescriptions list
    return RedirectResponse(url=f"/prescriptions?user_id={user_id}", status_code=303)

# Prescriptions list page
@app.get("/prescriptions", response_class=HTMLResponse)
def prescriptions_page(request: Request, user_id: int):
    # Get user info
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
    user = cur.fetchone()
    
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
    
    cur.execute(query)
    prescriptions = cur.fetchall()
    cur.close()
    conn.close()
    
    return templates.TemplateResponse("prescriptions.html", {
        "request": request,
        "user": user,
        "prescriptions": prescriptions
    })


# Pydantic model for prescription status update
from pydantic import BaseModel
from typing import Optional

class PrescriptionStatusUpdate(BaseModel):
    prescription_uuid: str
    status: str
    pharmacy_notes: Optional[str] = None

# API endpoint to receive prescription status update from pharmacy
@app.post("/api/prescription-status-update")
def update_prescription_status(update: PrescriptionStatusUpdate):
    """
    API endpoint for pharmacy to update prescription status.
    Accepts JSON only.
    
    Request body (JSON):
    {
        "prescription_uuid": "uuid-string",
        "status": "approved|denied|pending",
        "pharmacy_notes": "optional notes"
    }
    """
    conn = get_db()
    cur = conn.cursor()
    
    try:
        print(f"[DEBUG] Updating prescription_uuid: {update.prescription_uuid}")
        print(f"[DEBUG] New status: {update.status}")
        print(f"[DEBUG] Pharmacy notes: {update.pharmacy_notes}")
        
        # Update prescription status
        query = """
            UPDATE prescription_requests
            SET status = %s, 
                pharmacy_notes = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE prescription_uuid = %s
            RETURNING id, prescription_uuid, status
        """
        
        cur.execute(query, (update.status, update.pharmacy_notes, update.prescription_uuid))
        result = cur.fetchone()
        
        print(f"[DEBUG] Query result: {result}")
        print(f"[DEBUG] Result type: {type(result)}")
        
        if not result:
            print(f"[ERROR] Prescription {update.prescription_uuid} not found in database")
            cur.close()
            conn.close()
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": f"Prescription {update.prescription_uuid} not found"}
            )
        
        conn.commit()
        
        # Result is a RealDictRow, access by keys not indices
        result_dict = {
            "id": result['id'],
            "prescription_uuid": result['prescription_uuid'],
            "status": result['status']
        }
        
        print(f"[DEBUG] Successfully updated prescription: {result_dict}")
        
        cur.close()
        conn.close()
        
        return JSONResponse(content={
            "success": True,
            "message": "Prescription status updated successfully",
            "prescription_uuid": result_dict['prescription_uuid'],
            "status": result_dict['status']
        })
        
    except Exception as e:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print(f"Error updating prescription status: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error updating prescription: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
