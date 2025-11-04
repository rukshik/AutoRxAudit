"""
QKD Service - Quantum Key Distribution Service using BB84 Protocol
This service simulates a QKD system using the BB84 protocol for educational purposes.
"""

import os
import uuid
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from bb84 import simulate_qkd_exchange

# Load environment variables
load_dotenv()

# Configuration
QKD_SERVICE_PORT = int(os.getenv('QKD_SERVICE_PORT', 8005))
QKD_SERVICE_HOST = os.getenv('QKD_SERVICE_HOST', '0.0.0.0')
QKD_INITIAL_QUBITS = int(os.getenv('QKD_INITIAL_QUBITS', 512))
QKD_TARGET_KEY_BITS = int(os.getenv('QKD_TARGET_KEY_BITS', 256))
QKD_SESSION_TIMEOUT_SECONDS = int(os.getenv('QKD_SESSION_TIMEOUT_SECONDS', 300))
QKD_ENABLE_EAVESDROP_SIMULATION = os.getenv('QKD_ENABLE_EAVESDROP_SIMULATION', 'false').lower() == 'true'

# Initialize FastAPI app
app = FastAPI(
    title="QKD Service",
    description="Quantum Key Distribution Service using BB84 Protocol",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active sessions
# In production, use Redis or similar
active_sessions: Dict[str, Dict] = {}


# Pydantic models
class QKDInitiateRequest(BaseModel):
    sender: str  # e.g., "doctor-office-1"
    receiver: str  # e.g., "pharmacy-A"
    prescription_uuid: str


class QKDInitiateResponse(BaseModel):
    session_id: str
    status: str
    message: str


class QKDExchangeRequest(BaseModel):
    session_id: str


class QKDExchangeRequest(BaseModel):
    session_id: str
    party: str  # "sender" or "receiver" - identifies who is calling


class QKDExchangeResponse(BaseModel):
    session_id: str
    success: bool
    error_rate: float
    secure: bool
    sifted_bits: int
    final_key_bits: int
    key: str  # The key for THIS party (sender or receiver)
    key_id: str
    message: str


class QKDSessionStatus(BaseModel):
    session_id: str
    status: str
    sender: str
    receiver: str
    prescription_uuid: str
    created_at: str
    expires_at: str
    key_established: bool
    secure: bool
    error_rate: Optional[float]


# Helper functions
def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"qkd-{uuid.uuid4().hex[:16]}"


def generate_key_id(session_id: str) -> str:
    """Generate key ID from session ID"""
    return f"key-{hashlib.sha256(session_id.encode()).hexdigest()[:16]}"


def cleanup_expired_sessions():
    """Remove expired sessions from memory"""
    now = datetime.now()
    expired = [
        sid for sid, session in active_sessions.items()
        if datetime.fromisoformat(session['expires_at']) < now
    ]
    for sid in expired:
        del active_sessions[sid]
        print(f"Cleaned up expired session: {sid}")


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    cleanup_expired_sessions()
    return {
        "service": "QKD Service",
        "version": "1.0.0",
        "status": "running",
        "protocol": "BB84",
        "active_sessions": len(active_sessions)
    }


@app.post("/qkd/initiate", response_model=QKDInitiateResponse)
async def initiate_qkd_session(request: QKDInitiateRequest):
    """
    Initiate a new QKD session between sender and receiver
    
    This creates a session record but doesn't run the key exchange yet.
    """
    cleanup_expired_sessions()
    
    session_id = generate_session_id()
    created_at = datetime.now()
    expires_at = created_at + timedelta(seconds=QKD_SESSION_TIMEOUT_SECONDS)
    
    # Create session record (METADATA ONLY - no keys stored)
    active_sessions[session_id] = {
        'session_id': session_id,
        'sender': request.sender,
        'receiver': request.receiver,
        'prescription_uuid': request.prescription_uuid,
        'status': 'initiated',
        'created_at': created_at.isoformat(),
        'expires_at': expires_at.isoformat(),
        'key_established': False,
        'error_rate': None,
        'secure': None,
        'sender_retrieved': False,
        'receiver_retrieved': False
    }

    return QKDInitiateResponse(
        session_id=session_id,
        status='initiated',
        message='QKD session initiated. Call /qkd/exchange to complete key exchange.'
    )


@app.post("/qkd/exchange", response_model=QKDExchangeResponse)
async def complete_qkd_exchange(request: QKDExchangeRequest):
    """
    Complete the BB84 quantum key exchange
    
    This runs the full BB84 protocol and returns the key directly to the calling party.
    In real QKD, both parties derive the key simultaneously from the quantum channel.
    Here we simulate that by having each party call this endpoint once to "receive" their key.
    
    SECURITY: Keys are NEVER stored in the service - only returned directly to parties.
    """
    session_id = request.session_id
    party = request.party.lower()
    
    if party not in ['sender', 'receiver']:
        raise HTTPException(status_code=400, detail="Party must be 'sender' or 'receiver'")
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Check if session expired
    if datetime.fromisoformat(session['expires_at']) < datetime.now():
        del active_sessions[session_id]
        raise HTTPException(status_code=410, detail="Session expired")
    
    # Check if this party already retrieved their key
    retrieval_flag = f"{party}_retrieved"
    if session.get(retrieval_flag):
        raise HTTPException(status_code=400, detail=f"{party.capitalize()} already retrieved key for this session")
    
    # If first party to call, run BB84 protocol
    if not session['key_established']:
        print(f"Running BB84 Protocol for session: {session_id}")
        print(f"First party ({party}) triggering key exchange")
        
        # Run BB84 protocol
        result = simulate_qkd_exchange(
            n_qubits=QKD_INITIAL_QUBITS,
            target_bits=QKD_TARGET_KEY_BITS,
            simulate_eve=QKD_ENABLE_EAVESDROP_SIMULATION
        )
        
        # Store ONLY metadata (no keys!)
        session['status'] = 'completed' if result['success'] else 'failed'
        session['key_established'] = result['success']
        session['error_rate'] = result['error_rate']
        session['secure'] = result['secure']
        session['sifted_bits'] = result['sifted_bits']
        session['final_key_bits'] = result['final_key_bits']
        
        # Store keys TEMPORARILY in memory for second party retrieval
        # In real QKD, this wouldn't exist - both parties derive simultaneously
        session['_temp_alice_key'] = result.get('alice_key')
        session['_temp_bob_key'] = result.get('bob_key')
        session['_temp_result'] = result
        
        print(f"BB84 Protocol completed for session: {session_id}, success: {result['success']}, error_rate: {result['error_rate']:.4f}, Sifted: {result['sifted_bits']} bits")
    else:
        print(f"Session {session_id}: Second party ({party}) retrieving key")
        result = session.get('_temp_result', {})
    
    if not session['key_established']:
        raise HTTPException(status_code=500, detail="Key exchange failed - eavesdropping detected or protocol error")
    
    # Return appropriate key for this party
    if party == 'sender':
        key = session['_temp_alice_key']
    else:  # receiver
        key = session['_temp_bob_key']
    
    # Mark this party as having retrieved their key
    session[retrieval_flag] = True
    
    # If BOTH parties retrieved, wipe temporary keys from memory
    if session.get('sender_retrieved') and session.get('receiver_retrieved'):
        session.pop('_temp_alice_key', None)
        session.pop('_temp_bob_key', None)
        session.pop('_temp_result', None)
        print(f"Both parties retrieved keys - wiped temporary storage for session {session_id}")
    
    return QKDExchangeResponse(
        session_id=session_id,
        success=session['key_established'],
        error_rate=session['error_rate'],
        secure=session['secure'],
        sifted_bits=session['sifted_bits'],
        final_key_bits=session['final_key_bits'],
        key=key,  # Return key directly to THIS party
        key_id=generate_key_id(session_id),
        message=result.get('message', 'Key exchange successful')
    )


@app.get("/qkd/session/{session_id}", response_model=QKDSessionStatus)
async def get_session_status(session_id: str):
    """Get status of a QKD session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    return QKDSessionStatus(
        session_id=session['session_id'],
        status=session['status'],
        sender=session['sender'],
        receiver=session['receiver'],
        prescription_uuid=session['prescription_uuid'],
        created_at=session['created_at'],
        expires_at=session['expires_at'],
        key_established=session['key_established'],
        secure=session['secure'] if session['secure'] is not None else False,
        error_rate=session['error_rate']
    )


@app.delete("/qkd/destroy/{session_id}")
async def destroy_session(session_id: str):
    """
    Destroy a QKD session and wipe keys from memory
    
    Should be called after both parties have used the keys for encryption/decryption.
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Wipe any temporary keys still in memory (shouldn't exist if both parties retrieved)
    session.pop('_temp_alice_key', None)
    session.pop('_temp_bob_key', None)
    session.pop('_temp_result', None)
    
    # Remove session
    del active_sessions[session_id]
    
    print(f"Session destroyed: {session_id}")
    
    return {
        'session_id': session_id,
        'status': 'destroyed',
        'message': 'Session and keys securely destroyed'
    }


@app.get("/qkd/statistics")
async def get_statistics():
    """Get QKD service statistics"""
    cleanup_expired_sessions()
    
    total_sessions = len(active_sessions)
    completed_sessions = sum(1 for s in active_sessions.values() if s['status'] == 'completed')
    secure_sessions = sum(1 for s in active_sessions.values() if s.get('secure', False))
    
    return {
        'total_active_sessions': total_sessions,
        'completed_sessions': completed_sessions,
        'secure_sessions': secure_sessions,
        'protocol': 'BB84',
        'initial_qubits': QKD_INITIAL_QUBITS,
        'target_key_bits': QKD_TARGET_KEY_BITS,
        'session_timeout_seconds': QKD_SESSION_TIMEOUT_SECONDS
    }


if __name__ == "__main__":
    import uvicorn
    
    print(f"\n{'='*70}")
    print(f"   QKD SERVICE - EDUCATIONAL SIMULATION")
    print(f"{'='*70}")
    print(f"")
    print(f"   SECURITY NOTICE: This is NOT quantum-secure in practice")
    print(f"")
    print(f"   Why? Centralized architecture with key transmission over HTTPS")
    print(f"   Real QKD requires: Local quantum devices + fiber optic channel")
    print(f"")
    print(f"   This demo shows:")
    print(f"   BB84 protocol implementation (prepare, measure, sift)")
    print(f"   Eavesdropping detection via QBER")
    print(f"   Application integration patterns")
    print(f"")
    print(f"   For production: Deploy local QKD hardware at each endpoint")
    print(f"   See SECURITY.md for details")
    print(f"{'='*70}")
    print(f"")
    print(f"Configuration:")
    print(f"  Protocol: BB84 (Quantum Key Distribution)")
    print(f"  Port: {QKD_SERVICE_PORT}")
    print(f"  Initial Qubits: {QKD_INITIAL_QUBITS}")
    print(f"  Target Key Bits: {QKD_TARGET_KEY_BITS}")
    print(f"  Session Timeout: {QKD_SESSION_TIMEOUT_SECONDS}s")
    print(f"  Eavesdrop Simulation: {QKD_ENABLE_EAVESDROP_SIMULATION}")
    print(f"{'='*70}\n")
    
    uvicorn.run(
        "app:app",
        host=QKD_SERVICE_HOST,
        port=QKD_SERVICE_PORT,
        reload=True
    )
