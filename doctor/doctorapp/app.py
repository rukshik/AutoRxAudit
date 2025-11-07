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
BLOCKCHAIN_URL = os.getenv('BLOCKCHAIN_URL'. 'http://localhost:8001')

app = FastAPI(title="Doctor Office")

# Templates
templates = Jinja2Templates(directory="templates")

# Database connection
def get_db():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

# encrypt data using qkd key - AES encryption
def encrypt_with_qkd_key(data: dict, qkd_key_hex: str) -> str:
    
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
                # Initiate QKD session
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
                
                # Complete BB84 key exchange and receive key directly
                # In real QKD,  key is derived from the quantum channel
                qkd_exchange_response = await client.post(
                    f"{QKD_SERVICE_URL}/qkd/exchange",
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
                
                qkd_key = qkd_exchange_data['key']

                # Encrypt prescription data with quantum-safe key
                encrypted_payload = encrypt_with_qkd_key(prescription_data_to_send, qkd_key)
                
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
            print(f"QKD encryption failed: {qkd_err}")
    
    # Log to blockchain FIRST (before sending to pharmacy)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            blockchain_response = await client.post(
                f"${BLOCKCHAIN_URL}/pharmacy/prescription-created",
                json={
                    "prescription_uuid": prescription_uuid,
                    "doctor_id": str(user_id)
                }
            )
    except Exception as bc_err:
        print(f"Blockchain logging error: {bc_err}")
    
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
    
    # Destroy QKD key (if QKD was used)
    if qkd_session_id:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                destroy_response = await client.delete(
                    f"{QKD_SERVICE_URL}/qkd/destroy/{qkd_session_id}"
                )
              
        except Exception as destroy_err:
            print(f"QKD key destruction error: {destroy_err}")
    
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

# log doctor/pharamcy communication to a db a table (just for checking)
def log_communication(conn, prescription_uuid: str, 
                     action_type: str, actor_type: str, actor_id: int, 
                     actor_name: str, comments: str, previous_status: str, new_status: str):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO prescription_communications 
            (prescription_uuid, action_type, actor_type, 
             actor_id, actor_name, comments, previous_status, new_status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """, (prescription_uuid, action_type, actor_type, 
              actor_id, actor_name, comments, previous_status, new_status))
        conn.commit()
        print(f"✓ Communication logged: {action_type} by {actor_name}")
    except Exception as e:
        print(f"⚠️  Failed to log communication: {e}")
        conn.rollback()
    finally:
        cursor.close()

# API endpoint to receive prescription status update from pharmacy
@app.post("/api/prescription-status-update")
async def update_prescription_status(request: Request):
    conn = get_db()
    cur = conn.cursor()
    
    try:
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
        cur.execute("SELECT status FROM prescription_requests WHERE prescription_uuid = %s", (prescription_uuid,))
        current = cur.fetchone()
        previous_status = current['status'] if current else 'unknown'
        
        cur.execute(query, (status, pharmacy_notes, prescription_uuid))
        result = cur.fetchone()
        
        if not result:
            print(f"[ERROR] Prescription {prescription_uuid} not found in database")
            cur.close()
            conn.close()
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": f"Prescription {prescription_uuid} not found"}
            )
        
        conn.commit()
        
        # Log communication from pharmacy 
        if status == 'pending_review' and pharmacy_notes:
            log_communication(
                conn, prescription_uuid,
                'PHARMACIST_REQUEST_REVIEW', 'PHARMACIST', 0,
                "Pharmacist", pharmacy_notes,
                previous_status, status
            )
        
        result_dict = {
            "id": result['id'],
            "prescription_uuid": result['prescription_uuid'],
            "status": result['status']
        }
              
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

# Prescription review detail page
@app.get("/prescription/{prescription_id}/review", response_class=HTMLResponse)
def prescription_review_page(request: Request, prescription_id: int, user_id: int):
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Get user info
        cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
        user = cur.fetchone()
        
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
        cur.execute(query, (prescription_id, user_id))
        prescription = cur.fetchone()
        
        if not prescription:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        # Get communication history
        cur.execute("""
            SELECT * FROM prescription_communications
            WHERE prescription_uuid = %s
            ORDER BY created_at ASC
        """, (prescription['prescription_uuid'],))
        communications = cur.fetchall()
        
        if communications:
            for i, comm in enumerate(communications):
                print(f"  Communication {i+1}: {comm.get('action_type')} - {comm.get('comments', 'No comments')[:50]}")
        print()
        
        cur.close()
        conn.close()
        
        return templates.TemplateResponse("prescription_review.html", {
            "request": request,
            "user": user,
            "prescription": prescription,
            "communications": communications
        })
        
    except Exception as e:
        if cur:
            cur.close()
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=str(e))

# Get prescriptions under review
@app.get("/api/prescriptions-under-review")
def get_prescriptions_under_review(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    
    try:
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
        
        cur.execute(query, (user_id,))
        prescriptions = cur.fetchall()
        cur.close()
        conn.close()
        
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
        
    except Exception as e:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print(f"Error in get_prescriptions_under_review: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error fetching prescriptions: {str(e)}"}
        )

# Respond to review request
@app.post("/prescription/{prescription_id}/respond")
async def respond_to_review(
    request: Request,
    prescription_id: int,
    user_id: int = Form(...),
    response_action: str = Form(...),  # 'respond' or 'cancel'
    response_comments: str = Form(None)
):
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Get prescription details
        cur.execute(
            "SELECT * FROM prescription_requests WHERE id = %s AND prescriber_id = %s",
            (prescription_id, user_id)
        )
        prescription = cur.fetchone()
        
        if not prescription:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        # Get doctor details
        cur.execute("SELECT full_name FROM users WHERE user_id = %s", (user_id,))
        doctor = cur.fetchone()
        doctor_name = doctor['full_name'] if doctor else f"Doctor {user_id}"
        
        previous_status = prescription['status']
        
        if response_action == 'cancel':
            # Cancel the prescription
            new_status = 'cancelled'
            
            cur.execute("""
                UPDATE prescription_requests
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (new_status, prescription_id))
            
            conn.commit()
            
            # Log communication
            log_communication(
                conn, prescription['prescription_uuid'],
                'DOCTOR_CANCEL', 'DOCTOR', user_id,
                doctor_name, response_comments or "Prescription cancelled", 
                previous_status, new_status
            )
            
            # Log to blockchain
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    blockchain_response = await client.post(
                        f"{BLOCKCHAIN_URL}/doctor/cancel-prescription",
                        json={
                            "prescription_uuid": prescription['prescription_uuid'],
                            "doctor_id": str(user_id),
                            "cancellation_reason": response_comments or "Cancelled by doctor"
                        }
                    )
                    
            except Exception as bc_err:
                print(f"Blockchain cancellation logging error: {bc_err}")
            
            # Notify pharmacy
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{PHARMACY_API_URL}/api/prescription-status-update",
                        json={
                            "prescription_uuid": prescription['prescription_uuid'],
                            "status": "CANCELLED",
                            "pharmacy_notes": response_comments or "Cancelled by doctor"
                        }
                    )
            except Exception as e:
                print(f"Error notifying pharmacy: {e}")
                
        else:  # respond
            # Update status to under_review
            new_status = 'under_review'
            
            cur.execute("""
                UPDATE prescription_requests
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (new_status, prescription_id))
            
            conn.commit()
            
            # Log communication
            log_communication(
                conn, prescription['prescription_uuid'],
                'DOCTOR_RESPONSE', 'DOCTOR', user_id,
                doctor_name, response_comments or "", 
                previous_status, new_status
            )
            
            # Log to blockchain
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    blockchain_response = await client.post(
                        f"{BLOCKCHAIN_URL}/doctor/respond-to-review",
                        json={
                            "prescription_uuid": prescription['prescription_uuid'],
                            "doctor_id": str(user_id),
                            "doctor_name": doctor_name,
                            "response_comments": response_comments or ""
                        }
                    )
                 
            except Exception as bc_err:
                print(f"Blockchain response logging error: {bc_err}")
            
            # Notify pharmacy
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{PHARMACY_API_URL}/api/prescription-status-update",
                        json={
                            "prescription_uuid": prescription['prescription_uuid'],
                            "status": "UNDER_REVIEW",
                            "pharmacy_notes": response_comments or ""
                        }
                    )
            except Exception as e:
                print(f"Error notifying pharmacy: {e}")
        
        cur.close()
        conn.close()
        
        return RedirectResponse(
            url=f"/prescriptions?user_id={user_id}",
            status_code=303
        )
        
    except Exception as e:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print(f"Error responding to review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
