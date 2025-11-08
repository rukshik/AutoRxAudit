"""
Blockchain service for AutoRxAudit, both doctor and pharmacy app calls thi
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
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
account = None
pharmacy_contract = None  # Add pharmacy contract to globals

# intitalize blockchain
def initialize_blockchain() -> bool:
    global w3, account, pharmacy_contract

    try:
        w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC_URL))

        #load contract ABI
        pharmacy_contract = w3.eth.contract(
                address=Web3.to_checksum_address(PHARMACY_WORKFLOW_CONTRACT_ADDRESS),
                abi=PHARMACY_WORKFLOW_ABI
        )
        
        if DEPLOYER_PRIVATE_KEY:
            account = Account.from_key(DEPLOYER_PRIVATE_KEY)
        else:
            # Use first account from Hardhat
            account = w3.eth.accounts[0] if w3.eth.accounts else None
        print (f"Using account: {account if account else 'None'}")
  
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
# Log presecription created
@app.post("/pharmacy/prescription-created")
async def log_prescription_created(request: Request) -> Dict[str, Any]:
    
    try:
        data = await request.json()
        
        # Build transaction
        tx = pharmacy_contract.functions.logPrescriptionCreated(
            data['prescription_uuid'],
            data['doctor_id']
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
        print(f"Error logging prescription creation: {e} ")
        raise HTTPException(status_code=500, detail=f"Failed to log prescription creation: {str(e)}")

# AI review completed
@app.post("/pharmacy/ai-review")
async def log_ai_review(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        
        # Build transaction
        tx = pharmacy_contract.functions.logAIReview(
            data['prescription_uuid'],
            data['flagged'],
            data['eligibility_score'],
            data['oud_risk_score'],
            data['flag_reason'],
            data['recommendation']
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
async def log_pharmacist_decision(request: Request) -> Dict[str, Any]:
    
    try:
        data = await request.json()
        
        # Build transaction
        tx = pharmacy_contract.functions.logPharmacistDecision(
            data['prescription_uuid'],
            data['pharmacist_id'],
            data['action'],
            data['action_reason']
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
async def log_pharmacist_requests_review(request: Request) -> Dict[str, Any]: 
    try:
        data = await request.json()
        
        # Build transaction
        tx = pharmacy_contract.functions.logPharmacistRequestsReview(
            data['prescription_uuid'],
            data['pharmacist_id'],
            data['pharmacist_name'],
            data['review_comments']
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
async def log_doctor_responds_to_review(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        
        # Build transaction
        tx = pharmacy_contract.functions.logDoctorRespondsToReview(
            data['prescription_uuid'],
            data['doctor_id'],
            data.get('doctor_name', ''),
            data['response_comments']
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
async def log_doctor_cancels_prescription(request: Request) -> Dict[str, Any]:   
    try:
        data = await request.json()
        
        # Build transaction
        tx = pharmacy_contract.functions.logDoctorCancelsPrescription(
            data['prescription_uuid'],
            data['doctor_id'],
            data['cancellation_reason']
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

# intialize blockchain on startup
@app.on_event("startup")
async def startup_event():
    initialize_blockchain()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
