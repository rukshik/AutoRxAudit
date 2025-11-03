"""
Blockchain Microservice for AutoRxAudit
Handles immutable recording of flagged prescriptions and pharmacist actions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from web3 import Web3
from eth_account import Account
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Configuration from .env
BLOCKCHAIN_RPC_URL = os.getenv('BLOCKCHAIN_RPC_URL', 'http://127.0.0.1:8545')
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS', '')
DEPLOYER_PRIVATE_KEY = os.getenv('DEPLOYER_PRIVATE_KEY', '')
SERVICE_PORT = int(os.getenv('BLOCKCHAIN_SERVICE_PORT', '8001'))

# Initialize FastAPI
app = FastAPI(
    title="AutoRxAudit Blockchain Service",
    description="Immutable audit trail for prescription flagging and pharmacist actions",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Contract ABI (simplified - only what we need)
CONTRACT_ABI = [
    {
        "inputs": [
            {"name": "patientId", "type": "string"},
            {"name": "doctorId", "type": "string"},
            {"name": "pharmacyId", "type": "string"},
            {"name": "drugName", "type": "string"},
            {"name": "quantity", "type": "uint256"},
            {"name": "riskScore", "type": "uint8"},
            {"name": "riskFactors", "type": "string"}
        ],
        "name": "recordFlaggedPrescription",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "prescriptionId", "type": "uint256"},
            {"name": "overrideReason", "type": "string"},
            {"name": "overrideBy", "type": "string"}
        ],
        "name": "overridePrescription",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "", "type": "uint256"}],
        "name": "prescriptions",
        "outputs": [
            {"name": "prescriptionId", "type": "uint256"},
            {"name": "patientId", "type": "string"},
            {"name": "doctorId", "type": "string"},
            {"name": "pharmacyId", "type": "string"},
            {"name": "drugName", "type": "string"},
            {"name": "quantity", "type": "uint256"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "riskScore", "type": "uint8"},
            {"name": "riskFactors", "type": "string"},
            {"name": "isFlagged", "type": "bool"},
            {"name": "isOverridden", "type": "bool"},
            {"name": "overrideReason", "type": "string"},
            {"name": "overrideBy", "type": "string"},
            {"name": "overrideTimestamp", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "prescriptionCounter",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "prescriptionId", "type": "uint256"},
            {"indexed": False, "name": "patientId", "type": "string"},
            {"indexed": False, "name": "doctorId", "type": "string"},
            {"indexed": False, "name": "pharmacyId", "type": "string"},
            {"indexed": False, "name": "riskScore", "type": "uint8"},
            {"indexed": False, "name": "riskFactors", "type": "string"}
        ],
        "name": "PrescriptionFlagged",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "prescriptionId", "type": "uint256"},
            {"indexed": False, "name": "overrideBy", "type": "string"},
            {"indexed": False, "name": "overrideReason", "type": "string"},
            {"indexed": False, "name": "timestamp", "type": "uint256"}
        ],
        "name": "PrescriptionOverridden",
        "type": "event"
    }
]

# Initialize Web3 and contract
w3 = None
contract = None
account = None

def initialize_blockchain():
    """Initialize blockchain connection"""
    global w3, contract, account
    
    if not CONTRACT_ADDRESS:
        print("⚠️  CONTRACT_ADDRESS not set in .env - run deployment first!")
        return False
    
    try:
        w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC_URL))
        
        if not w3.is_connected():
            print(f"❌ Cannot connect to blockchain at {BLOCKCHAIN_RPC_URL}")
            return False
        
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
        
        if DEPLOYER_PRIVATE_KEY:
            account = Account.from_key(DEPLOYER_PRIVATE_KEY)
        else:
            # Use first account from Hardhat
            account = w3.eth.accounts[0] if w3.eth.accounts else None
        
        print(f"✓ Connected to blockchain at {BLOCKCHAIN_RPC_URL}")
        print(f"✓ Contract address: {CONTRACT_ADDRESS}")
        print(f"✓ Using account: {account if isinstance(account, str) else account.address if account else 'None'}")
        return True
    except Exception as e:
        print(f"❌ Blockchain initialization failed: {e}")
        return False

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FlaggedPrescriptionRequest(BaseModel):
    """Request to record a flagged prescription on blockchain"""
    patient_id: str
    drug_name: str
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    risk_factors: str = Field(..., description="JSON string of risk factors")
    doctor_id: str = "SYSTEM"
    pharmacy_id: str = "PENDING"
    quantity: int = 1

class PharmacistActionRequest(BaseModel):
    """Request to record pharmacist action on blockchain"""
    blockchain_prescription_id: int
    action: str  # APPROVED, DENIED, OVERRIDE_APPROVE, OVERRIDE_DENY
    action_reason: str
    pharmacist_id: str

class BlockchainRecordResponse(BaseModel):
    """Response with blockchain transaction details"""
    success: bool
    blockchain_prescription_id: Optional[int] = None
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    message: str

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    is_connected = w3 and w3.is_connected() if w3 else False
    return {
        "status": "healthy" if is_connected else "degraded",
        "service": "Blockchain Service",
        "blockchain_connected": is_connected,
        "rpc_url": BLOCKCHAIN_RPC_URL,
        "contract_address": CONTRACT_ADDRESS if CONTRACT_ADDRESS else "NOT_CONFIGURED"
    }

@app.post("/record-flagged-prescription", response_model=BlockchainRecordResponse)
async def record_flagged_prescription(request: FlaggedPrescriptionRequest):
    """Record a flagged prescription to blockchain (immutable audit trail)"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        # Build transaction
        if isinstance(account, str):
            # Using Hardhat account (no private key needed)
            tx_hash = contract.functions.recordFlaggedPrescription(
                request.patient_id,
                request.doctor_id,
                request.pharmacy_id,
                request.drug_name,
                request.quantity,
                request.risk_score,
                request.risk_factors
            ).transact({'from': account})
        else:
            # Using private key account
            tx = contract.functions.recordFlaggedPrescription(
                request.patient_id,
                request.doctor_id,
                request.pharmacy_id,
                request.drug_name,
                request.quantity,
                request.risk_score,
                request.risk_factors
            ).build_transaction({
                'from': account.address,
                'nonce': w3.eth.get_transaction_count(account.address),
                'gas': 2000000,
                'gasPrice': w3.eth.gas_price
            })
            
            signed_tx = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get prescription ID from contract
        prescription_counter = contract.functions.prescriptionCounter().call()
        
        return {
            "success": True,
            "blockchain_prescription_id": prescription_counter,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": f"Flagged prescription recorded on blockchain (ID: {prescription_counter})"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain recording failed: {str(e)}")

@app.post("/record-pharmacist-action", response_model=BlockchainRecordResponse)
async def record_pharmacist_action(request: PharmacistActionRequest):
    """Record pharmacist action (override) to blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        # Build transaction
        if isinstance(account, str):
            tx_hash = contract.functions.overridePrescription(
                request.blockchain_prescription_id,
                request.action_reason,
                request.pharmacist_id
            ).transact({'from': account})
        else:
            tx = contract.functions.overridePrescription(
                request.blockchain_prescription_id,
                request.action_reason,
                request.pharmacist_id
            ).build_transaction({
                'from': account.address,
                'nonce': w3.eth.get_transaction_count(account.address),
                'gas': 2000000,
                'gasPrice': w3.eth.gas_price
            })
            
            signed_tx = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": True,
            "blockchain_prescription_id": request.blockchain_prescription_id,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": f"Pharmacist action recorded on blockchain"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain recording failed: {str(e)}")

@app.get("/prescription/{blockchain_id}")
async def get_prescription(blockchain_id: int):
    """Get prescription record from blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        prescription = contract.functions.prescriptions(blockchain_id).call()
        
        return {
            "prescription_id": prescription[0],
            "patient_id": prescription[1],
            "doctor_id": prescription[2],
            "pharmacy_id": prescription[3],
            "drug_name": prescription[4],
            "quantity": prescription[5],
            "timestamp": prescription[6],
            "risk_score": prescription[7],
            "risk_factors": prescription[8],
            "is_flagged": prescription[9],
            "is_overridden": prescription[10],
            "override_reason": prescription[11],
            "override_by": prescription[12],
            "override_timestamp": prescription[13]
        }
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Prescription not found: {str(e)}")

@app.get("/prescriptions/count")
async def get_prescription_count():
    """Get total count of prescriptions on blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        count = contract.functions.prescriptionCounter().call()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")

@app.get("/prescriptions/all")
async def get_all_prescriptions(limit: int = 50):
    """Get all prescription records from blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        count = contract.functions.prescriptionCounter().call()
        prescriptions = []
        
        # Get last N prescriptions (most recent first)
        start = max(1, count - limit + 1)
        for i in range(count, start - 1, -1):
            try:
                prescription = contract.functions.prescriptions(i).call()
                prescriptions.append({
                    "prescription_id": prescription[0],
                    "patient_id": prescription[1],
                    "drug_name": prescription[4],
                    "risk_score": prescription[7],
                    "timestamp": prescription[6],
                    "is_flagged": prescription[9],
                    "is_overridden": prescription[10],
                    "override_by": prescription[12]
                })
            except:
                continue
        
        return {"prescriptions": prescriptions, "total_count": count}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prescriptions: {str(e)}")

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize blockchain connection on startup"""
    print("\n" + "="*60)
    print("Starting Blockchain Microservice...")
    print("="*60)
    initialize_blockchain()
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
