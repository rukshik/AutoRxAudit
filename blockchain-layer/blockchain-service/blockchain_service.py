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

# Contract ABI (from compiled artifacts)
CONTRACT_ABI = [
    {
        "inputs": [
            {
                "internalType": "struct PrescriptionAuditContract.AuditInput",
                "name": "input",
                "type": "tuple",
                "components": [
                    {"internalType": "uint256", "name": "auditId", "type": "uint256"},
                    {"internalType": "uint256", "name": "prescriptionId", "type": "uint256"},
                    {"internalType": "string", "name": "patientId", "type": "string"},
                    {"internalType": "string", "name": "drugName", "type": "string"},
                    {"internalType": "uint8", "name": "eligibilityScore", "type": "uint8"},
                    {"internalType": "uint8", "name": "eligibilityPrediction", "type": "uint8"},
                    {"internalType": "uint8", "name": "oudRiskScore", "type": "uint8"},
                    {"internalType": "uint8", "name": "oudRiskPrediction", "type": "uint8"},
                    {"internalType": "bool", "name": "flagged", "type": "bool"},
                    {"internalType": "string", "name": "flagReason", "type": "string"},
                    {"internalType": "string", "name": "recommendation", "type": "string"}
                ]
            }
        ],
        "name": "recordAudit",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "auditId", "type": "uint256"},
            {"internalType": "string", "name": "action", "type": "string"},
            {"internalType": "string", "name": "actionReason", "type": "string"},
            {"internalType": "string", "name": "reviewedBy", "type": "string"},
            {"internalType": "string", "name": "reviewedByName", "type": "string"},
            {"internalType": "string", "name": "reviewedByEmail", "type": "string"}
        ],
        "name": "recordPharmacistAction",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "blockchainId", "type": "uint256"}],
        "name": "getAuditRecord",
        "outputs": [
            {
                "internalType": "struct PrescriptionAuditContract.PrescriptionRecord",
                "name": "",
                "type": "tuple",
                "components": [
                    {"internalType": "uint256", "name": "blockchainId", "type": "uint256"},
                    {"internalType": "uint256", "name": "auditId", "type": "uint256"},
                    {"internalType": "uint256", "name": "prescriptionId", "type": "uint256"},
                    {"internalType": "string", "name": "patientId", "type": "string"},
                    {"internalType": "string", "name": "drugName", "type": "string"},
                    {"internalType": "uint8", "name": "eligibilityScore", "type": "uint8"},
                    {"internalType": "uint8", "name": "eligibilityPrediction", "type": "uint8"},
                    {"internalType": "uint8", "name": "oudRiskScore", "type": "uint8"},
                    {"internalType": "uint8", "name": "oudRiskPrediction", "type": "uint8"},
                    {"internalType": "bool", "name": "flagged", "type": "bool"},
                    {"internalType": "string", "name": "flagReason", "type": "string"},
                    {"internalType": "string", "name": "recommendation", "type": "string"},
                    {"internalType": "uint256", "name": "auditedAt", "type": "uint256"},
                    {"internalType": "string", "name": "reviewedBy", "type": "string"},
                    {"internalType": "string", "name": "reviewedByName", "type": "string"},
                    {"internalType": "string", "name": "reviewedByEmail", "type": "string"},
                    {"internalType": "string", "name": "action", "type": "string"},
                    {"internalType": "string", "name": "actionReason", "type": "string"},
                    {"internalType": "uint256", "name": "reviewedAt", "type": "uint256"}
                ]
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "auditId", "type": "uint256"}],
        "name": "getBlockchainIdByAuditId",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "recordCounter",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
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

class AuditRecordRequest(BaseModel):
    """Request to record audit result on blockchain (matches audit_logs schema)"""
    audit_id: int
    prescription_id: int
    patient_id: str
    drug_name: str
    eligibility_score: int = Field(..., ge=0, le=100)
    eligibility_prediction: int = Field(..., ge=0, le=1)
    oud_risk_score: int = Field(..., ge=0, le=100)
    oud_risk_prediction: int = Field(..., ge=0, le=1)
    flagged: bool
    flag_reason: str
    recommendation: str

class PharmacistActionRequest(BaseModel):
    """Request to record pharmacist action on blockchain"""
    audit_id: int
    action: str  # APPROVED, DENIED, OVERRIDE_APPROVE, OVERRIDE_DENY
    action_reason: str
    reviewed_by: str  # user_id as string
    reviewed_by_name: str  # reviewer's full name
    reviewed_by_email: str  # reviewer's email

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

@app.post("/record-audit", response_model=BlockchainRecordResponse)
async def record_audit(request: AuditRecordRequest):
    """Record audit result to blockchain (called after AI model evaluation)"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        # Build audit input tuple
        audit_input = (
            request.audit_id,
            request.prescription_id,
            request.patient_id,
            request.drug_name,
            request.eligibility_score,
            request.eligibility_prediction,
            request.oud_risk_score,
            request.oud_risk_prediction,
            request.flagged,
            request.flag_reason,
            request.recommendation
        )
        
        # Build transaction
        if isinstance(account, str):
            # Using Hardhat account (no private key needed)
            tx_hash = contract.functions.recordAudit(audit_input).transact({'from': account})
        else:
            # Using private key account
            tx = contract.functions.recordAudit(audit_input).build_transaction({
                'from': account.address,
                'nonce': w3.eth.get_transaction_count(account.address),
                'gas': 2000000,
                'gasPrice': w3.eth.gas_price
            })
            
            signed_tx = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get blockchain ID from contract
        blockchain_id = contract.functions.getBlockchainIdByAuditId(request.audit_id).call()
        
        return {
            "success": True,
            "blockchain_prescription_id": blockchain_id,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": f"Audit recorded on blockchain (Blockchain ID: {blockchain_id}, Audit ID: {request.audit_id})"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain recording failed: {str(e)}")

@app.post("/record-pharmacist-action", response_model=BlockchainRecordResponse)
async def record_pharmacist_action(request: PharmacistActionRequest):
    """Record pharmacist action to blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        # Build transaction
        if isinstance(account, str):
            tx_hash = contract.functions.recordPharmacistAction(
                request.audit_id,
                request.action,
                request.action_reason,
                request.reviewed_by,
                request.reviewed_by_name,
                request.reviewed_by_email
            ).transact({'from': account})
        else:
            tx = contract.functions.recordPharmacistAction(
                request.audit_id,
                request.action,
                request.action_reason,
                request.reviewed_by,
                request.reviewed_by_name,
                request.reviewed_by_email
            ).build_transaction({
                'from': account.address,
                'nonce': w3.eth.get_transaction_count(account.address),
                'gas': 2000000,
                'gasPrice': w3.eth.gas_price
            })
            
            signed_tx = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Decode return value from transaction logs to get new blockchain ID
        # The function now returns the new blockchain ID
        logs = contract.events.PharmacistActionRecorded().process_receipt(receipt)
        blockchain_id = logs[0]['args']['blockchainId'] if logs else None
        
        return {
            "success": True,
            "blockchain_prescription_id": blockchain_id,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": f"Pharmacist action recorded on blockchain (New Record ID: {blockchain_id})"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain recording failed: {str(e)}")

@app.get("/audit-record/{blockchain_id}")
async def get_audit_record(blockchain_id: int):
    """Get audit record from blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        record = contract.functions.getAuditRecord(blockchain_id).call()
        
        return {
            "blockchain_id": record[0],
            "audit_id": record[1],
            "prescription_id": record[2],
            "patient_id": record[3],
            "drug_name": record[4],
            "eligibility_score": record[5],
            "eligibility_prediction": record[6],
            "oud_risk_score": record[7],
            "oud_risk_prediction": record[8],
            "flagged": record[9],
            "flag_reason": record[10],
            "recommendation": record[11],
            "audited_at": record[12],
            "reviewed_by": record[13],
            "reviewed_by_name": record[14],
            "reviewed_by_email": record[15],
            "action": record[16],
            "action_reason": record[17],
            "reviewed_at": record[18]
        }
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Audit record not found: {str(e)}")

@app.get("/audit-records/count")
async def get_audit_record_count():
    """Get total count of audit records on blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        count = contract.functions.recordCounter().call()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")

@app.get("/audit-records/all")
async def get_all_audit_records(limit: int = 50):
    """Get all audit records from blockchain"""
    
    if not w3 or not contract:
        raise HTTPException(status_code=503, detail="Blockchain not initialized")
    
    try:
        count = contract.functions.recordCounter().call()
        records = []
        
        # Get last N records (most recent first)
        start = max(1, count - limit + 1)
        for i in range(count, start - 1, -1):
            try:
                record = contract.functions.getAuditRecord(i).call()
                records.append({
                    "blockchain_id": record[0],
                    "audit_id": record[1],
                    "prescription_id": record[2],
                    "patient_id": record[3],
                    "drug_name": record[4],
                    "eligibility_score": record[5],
                    "oud_risk_score": record[7],
                    "flagged": record[9],
                    "audited_at": record[12],
                    "reviewed_by": record[13],
                    "reviewed_by_name": record[14],
                    "action": record[16],
                    "reviewed_at": record[18]
                })
            except:
                continue
        
        return {"audit_records": records, "total_count": count}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch audit records: {str(e)}")

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
