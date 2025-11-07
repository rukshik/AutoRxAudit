"""
Blockchain service for AutoRxAudit, both doctor and pharmacy app calls thi
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
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Web3 and contract
w3 = None
contract = None
account = None
pharmacy_contract = None  # Add pharmacy contract to globals


# intitalize blockchain
def initialize_blockchain():
    global w3, contract, account, pharmacy_contract

    try:
        w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC_URL))

        # contract is no longer used. Onlu pharamcy contract is used    
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
        pharmacy_contract = w3.eth.contract(
                address=Web3.to_checksum_address(PHARMACY_WORKFLOW_CONTRACT_ADDRESS),
                abi=PHARMACY_WORKFLOW_ABI
        )
        
        if DEPLOYER_PRIVATE_KEY:
            account = Account.from_key(DEPLOYER_PRIVATE_KEY)
        else:
            # Use first account from Hardhat
            account = w3.eth.accounts[0] if w3.eth.accounts else None
  
        return True
    except Exception as e:
        print(f"Blockchain initialization failed: {e}")
        return False

# Load pharmacy workflow contract configuration
PHARMACY_WORKFLOW_CONTRACT_ADDRESS = os.getenv('PHARMACY_WORKFLOW_CONTRACT_ADDRESS', '')

# Pharmacy workflow contract ABI
PHARMACY_WORKFLOW_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"internalType": "string", "name": "doctorId", "type": "string"}
        ],
        "name": "logPrescriptionCreated",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"internalType": "bool", "name": "flagged", "type": "bool"},
            {"internalType": "uint8", "name": "eligibilityScore", "type": "uint8"},
            {"internalType": "uint8", "name": "oudRiskScore", "type": "uint8"},
            {"internalType": "string", "name": "flagReason", "type": "string"},
            {"internalType": "string", "name": "recommendation", "type": "string"}
        ],
        "name": "logAIReview",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"internalType": "string", "name": "pharmacistId", "type": "string"},
            {"internalType": "string", "name": "action", "type": "string"},
            {"internalType": "string", "name": "actionReason", "type": "string"}
        ],
        "name": "logPharmacistDecision",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"internalType": "string", "name": "pharmacistId", "type": "string"},
            {"internalType": "string", "name": "pharmacistName", "type": "string"},
            {"internalType": "string", "name": "reviewComments", "type": "string"}
        ],
        "name": "logPharmacistRequestsReview",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"internalType": "string", "name": "doctorId", "type": "string"},
            {"internalType": "string", "name": "doctorName", "type": "string"},
            {"internalType": "string", "name": "responseComments", "type": "string"}
        ],
        "name": "logDoctorRespondsToReview",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"internalType": "string", "name": "doctorId", "type": "string"},
            {"internalType": "string", "name": "cancellationReason", "type": "string"}
        ],
        "name": "logDoctorCancelsPrescription",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getStatistics",
        "outputs": [
            {"internalType": "uint256", "name": "prescriptions", "type": "uint256"},
            {"internalType": "uint256", "name": "aiReviews", "type": "uint256"},
            {"internalType": "uint256", "name": "pharmacistDecisions", "type": "uint256"},
            {"internalType": "uint256", "name": "reviewRequests", "type": "uint256"},
            {"internalType": "uint256", "name": "doctorResponses", "type": "uint256"},
            {"internalType": "uint256", "name": "cancellations", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "doctorId", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "PrescriptionCreated",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"indexed": False, "internalType": "bool", "name": "flagged", "type": "bool"},
            {"indexed": False, "internalType": "uint8", "name": "eligibilityScore", "type": "uint8"},
            {"indexed": False, "internalType": "uint8", "name": "oudRiskScore", "type": "uint8"},
            {"indexed": False, "internalType": "string", "name": "flagReason", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "recommendation", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "AIReviewCompleted",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "pharmacistId", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "action", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "actionReason", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "PharmacistDecision",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "pharmacistId", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "pharmacistName", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "reviewComments", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "PharmacistRequestsReview",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "doctorId", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "doctorName", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "responseComments", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "DoctorRespondsToReview",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "string", "name": "prescriptionUuid", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "doctorId", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "cancellationReason", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "DoctorCancelsPrescription",
        "type": "event"
    }
]

# Pydantic models for pharmacy workflow
# Prescription created (by doctor)
class PrescriptionCreatedRequest(BaseModel):
    prescription_uuid: str
    doctor_id: str

# AI review completed
class AIReviewRequest(BaseModel):
    prescription_uuid: str
    flagged: bool
    eligibility_score: int = Field(..., ge=0, le=100)
    oud_risk_score: int = Field(..., ge=0, le=100)
    flag_reason: str = ""
    recommendation: str

# Pharamacist decision
class PharmacistDecisionRequest(BaseModel):
    prescription_uuid: str
    pharmacist_id: str
    action: str  
    action_reason: str = ""

# Phramacist requested review
class PharmacistRequestsReviewRequest(BaseModel):
    prescription_uuid: str
    pharmacist_id: str
    pharmacist_name: str
    review_comments: str

# Dcotors response
class DoctorRespondsToReviewRequest(BaseModel):
    prescription_uuid: str
    doctor_id: str
    doctor_name: str
    response_comments: str

# Doctor cancels
class DoctorCancelsPrescriptionRequest(BaseModel):
    prescription_uuid: str
    doctor_id: str
    cancellation_reason: str

# Log presecription created
@app.post("/pharmacy/prescription-created")
async def log_prescription_created(request: PrescriptionCreatedRequest):
    
    try:
        # Build transaction
        tx = pharmacy_contract.functions.logPrescriptionCreated(
            request.prescription_uuid,
            request.doctor_id
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=DEPLOYER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": True,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": "Prescription creation logged to blockchain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log prescription creation: {str(e)}")

# AI review completed
@app.post("/pharmacy/ai-review")
async def log_ai_review(request: AIReviewRequest):
    try:
        # Build transaction
        tx = pharmacy_contract.functions.logAIReview(
            request.prescription_uuid,
            request.flagged,
            request.eligibility_score,
            request.oud_risk_score,
            request.flag_reason,
            request.recommendation
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=DEPLOYER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": True,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": "AI review logged to blockchain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log AI review: {str(e)}")

# phamacist decision
@app.post("/pharmacy/pharmacist-decision")
async def log_pharmacist_decision(request: PharmacistDecisionRequest):
    
    try:
        # Build transaction
        tx = pharmacy_contract.functions.logPharmacistDecision(
            request.prescription_uuid,
            request.pharmacist_id,
            request.action,
            request.action_reason
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=DEPLOYER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": True,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": "Pharmacist decision logged to blockchain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log pharmacist decision: {str(e)}")

# Pharamcist requested review
@app.post("/pharmacy/request-review")
async def log_pharmacist_requests_review(request: PharmacistRequestsReviewRequest): 
    try:
        # Build transaction
        tx = pharmacy_contract.functions.logPharmacistRequestsReview(
            request.prescription_uuid,
            request.pharmacist_id,
            request.pharmacist_name,
            request.review_comments
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=DEPLOYER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": True,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": "Pharmacist review request logged to blockchain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log review request: {str(e)}")

# Doctos response
@app.post("/doctor/respond-to-review")
async def log_doctor_responds_to_review(request: DoctorRespondsToReviewRequest):
    try:
        # Build transaction
        tx = pharmacy_contract.functions.logDoctorRespondsToReview(
            request.prescription_uuid,
            request.doctor_id,
            request.doctor_name,
            request.response_comments
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=DEPLOYER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": True,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": "Doctor response logged to blockchain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log doctor response: {str(e)}")

# Doctor cancels prescription
@app.post("/doctor/cancel-prescription")
async def log_doctor_cancels_prescription(request: DoctorCancelsPrescriptionRequest):   
    try:
        # Build transaction
        tx = pharmacy_contract.functions.logDoctorCancelsPrescription(
            request.prescription_uuid,
            request.doctor_id,
            request.cancellation_reason
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=DEPLOYER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "success": True,
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "message": "Prescription cancellation logged to blockchain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log cancellation: {str(e)}")

# Get full prescription trail
@app.get("/pharmacy/prescription-trail/{prescription_uuid}")
async def get_prescription_trail(prescription_uuid: str):
    if not w3 or not pharmacy_contract:
        raise HTTPException(status_code=503, detail="Pharmacy workflow contract not available")
    
    try:
        # Get all events for this prescription UUID
        prescription_created_filter = pharmacy_contract.events.PrescriptionCreated.create_filter(
            fromBlock=0,
            argument_filters={'prescriptionUuid': prescription_uuid}
        )
        ai_review_filter = pharmacy_contract.events.AIReviewCompleted.create_filter(
            fromBlock=0,
            argument_filters={'prescriptionUuid': prescription_uuid}
        )
        pharmacist_decision_filter = pharmacy_contract.events.PharmacistDecision.create_filter(
            fromBlock=0,
            argument_filters={'prescriptionUuid': prescription_uuid}
        )
        pharmacist_request_filter = pharmacy_contract.events.PharmacistRequestsReview.create_filter(
            fromBlock=0,
            argument_filters={'prescriptionUuid': prescription_uuid}
        )
        doctor_response_filter = pharmacy_contract.events.DoctorRespondsToReview.create_filter(
            fromBlock=0,
            argument_filters={'prescriptionUuid': prescription_uuid}
        )
        doctor_cancel_filter = pharmacy_contract.events.DoctorCancelsPrescription.create_filter(
            fromBlock=0,
            argument_filters={'prescriptionUuid': prescription_uuid}
        )
        
        # Fetch events
        created_events = prescription_created_filter.get_all_entries()
        ai_review_events = ai_review_filter.get_all_entries()
        decision_events = pharmacist_decision_filter.get_all_entries()
        request_review_events = pharmacist_request_filter.get_all_entries()
        doctor_response_events = doctor_response_filter.get_all_entries()
        doctor_cancel_events = doctor_cancel_filter.get_all_entries()
        
        # Build timeline
        timeline = []
        
        for event in created_events:
            timeline.append({
                "event_type": "prescription_created",
                "timestamp": event.args.timestamp,
                "doctor_id": event.args.doctorId,
                "block_number": event.blockNumber,
                "transaction_hash": event.transactionHash.hex()
            })
        
        for event in ai_review_events:
            timeline.append({
                "event_type": "ai_review",
                "timestamp": event.args.timestamp,
                "flagged": event.args.flagged,
                "eligibility_score": event.args.eligibilityScore,
                "oud_risk_score": event.args.oudRiskScore,
                "flag_reason": event.args.flagReason,
                "recommendation": event.args.recommendation,
                "block_number": event.blockNumber,
                "transaction_hash": event.transactionHash.hex()
            })
        
        for event in decision_events:
            timeline.append({
                "event_type": "pharmacist_decision",
                "timestamp": event.args.timestamp,
                "pharmacist_id": event.args.pharmacistId,
                "action": event.args.action,
                "action_reason": event.args.actionReason,
                "block_number": event.blockNumber,
                "transaction_hash": event.transactionHash.hex()
            })
        
        for event in request_review_events:
            timeline.append({
                "event_type": "pharmacist_requests_review",
                "timestamp": event.args.timestamp,
                "pharmacist_id": event.args.pharmacistId,
                "pharmacist_name": event.args.pharmacistName,
                "review_comments": event.args.reviewComments,
                "block_number": event.blockNumber,
                "transaction_hash": event.transactionHash.hex()
            })
        
        for event in doctor_response_events:
            timeline.append({
                "event_type": "doctor_responds_to_review",
                "timestamp": event.args.timestamp,
                "doctor_id": event.args.doctorId,
                "doctor_name": event.args.doctorName,
                "response_comments": event.args.responseComments,
                "block_number": event.blockNumber,
                "transaction_hash": event.transactionHash.hex()
            })
        
        for event in doctor_cancel_events:
            timeline.append({
                "event_type": "doctor_cancels_prescription",
                "timestamp": event.args.timestamp,
                "doctor_id": event.args.doctorId,
                "cancellation_reason": event.args.cancellationReason,
                "block_number": event.blockNumber,
                "transaction_hash": event.transactionHash.hex()
            })
        
        # Sort by block number (primary) and timestamp (secondary) to ensure correct chronological order
        timeline.sort(key=lambda x: (x['block_number'], x['timestamp']))
        
        return {
            "prescription_uuid": prescription_uuid,
            "total_events": len(timeline),
            "timeline": timeline
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prescription trail: {str(e)}")


#####################################################
# old contract no longer using
# Pedantic Models for API
# Audit log (not using anymore)

# Contract ABI 
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


class AuditRecordRequest(BaseModel):
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

# Pharmacist action
class PharmacistActionRequest(BaseModel):
    audit_id: int
    action: str 
    action_reason: str
    reviewed_by: str 
    reviewed_by_name: str 
    reviewed_by_email: str  

# Blockchain record response (all transactons)
class BlockchainRecordResponse(BaseModel):
    success: bool
    blockchain_prescription_id: Optional[int] = None
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    message: str

# Record audit log (not using it anymore)
@app.post("/record-audit", response_model=BlockchainRecordResponse)
async def record_audit(request: AuditRecordRequest):
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


# get audit record (no longer using)
@app.get("/audit-record/{blockchain_id}")
async def get_audit_record(blockchain_id: int):

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

# Record pharmacist action
@app.post("/record-pharmacist-action", response_model=BlockchainRecordResponse)
async def record_pharmacist_action(request: PharmacistActionRequest):
    
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
        
        #Blockchain id
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
==============================

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    initialize_blockchain()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
