"""
Blockchain Integration Module for AutoRxAudit
Handles prescription flagging, override tracking, and verification on blockchain
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from web3 import Web3
from eth_account import Account
import hashlib
import logging
from patient_data_handler import PatientDataHandler, load_patient_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PrescriptionData:
    """Data structure for prescription information"""

    patient_id: str
    doctor_id: str
    pharmacy_id: str
    drug_name: str
    quantity: int
    risk_score: int
    risk_factors: Dict[str, any]
    timestamp: Optional[int] = None


@dataclass
class OverrideData:
    """Data structure for prescription override information"""

    prescription_id: int
    override_reason: str
    override_by: str
    timestamp: Optional[int] = None


class BlockchainPrescriptionAudit:
    """
    Blockchain-based prescription auditing system
    Integrates with smart contract for tamper-proof record keeping
    """

    def __init__(
        self,
        config_file: str = "blockchain_config.json",
    ):
        """
        Initialize blockchain connection

        Args:
            config_file: Path to blockchain configuration file
        """
        # Load configuration
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.contract_address = config['contract_address']
            rpc_url = config['network']['provider']
            
            logger.info(f"Using RPC URL: {rpc_url}")
            logger.info(f"Using contract address: {self.contract_address}")
            
        except Exception as e:
            raise Exception(f"Failed to load blockchain config: {e}")

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        # Use the first account (same as deployment)
        self.w3.eth.default_account = self.w3.eth.accounts[0]
        self.account = self.w3.eth.default_account
        """
        Initialize blockchain connection

        Args:
            config_file: Path to blockchain configuration file
            private_key: Private key for signing transactions
            custom_provider: Optional custom RPC endpoint (overrides config)
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            rpc_url = config['network']['provider']
            self.contract_address = config['contract_address']
        except Exception as e:
            logger.error(f"Error loading blockchain config: {e}")
            raise

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        # Using default account from web3

        # Contract ABI
        self.contract_abi = [
            # Events
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
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "prescriptionId", "type": "uint256"},
                    {"indexed": False, "name": "verifiedBy", "type": "string"},
                    {"indexed": False, "name": "timestamp", "type": "uint256"}
                ],
                "name": "PrescriptionVerified",
                "type": "event"
            },
            # Functions
            {
                "inputs": [],
                "name": "owner",
                "outputs": [{"name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "doctor", "type": "address"}],
                "name": "addAuthorizedDoctor",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "pharmacist", "type": "address"}],
                "name": "addAuthorizedPharmacist",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "auditor", "type": "address"}],
                "name": "addAuthorizedAuditor",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
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
                "inputs": [
                    {"name": "prescriptionId", "type": "uint256"},
                    {"name": "verifiedBy", "type": "string"}
                ],
                "name": "verifyPrescription",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "prescriptionId", "type": "uint256"}],
                "name": "getPrescription",
                "outputs": [
                    {"name": "id", "type": "uint256"},
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
                "inputs": [{"name": "patientId", "type": "string"}],
                "name": "getPatientPrescriptions",
                "outputs": [{"name": "", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "doctorId", "type": "string"}],
                "name": "getDoctorPrescriptions",
                "outputs": [{"name": "", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "pharmacyId", "type": "string"}],
                "name": "getPharmacyPrescriptions",
                "outputs": [{"name": "", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getTotalPrescriptions",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "prescriptionId", "type": "uint256"}],
                "name": "isPrescriptionFlagged",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

        # Initialize contract if address provided
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=self.contract_abi,
            )
        else:
            self.contract = None
            logger.warning(
                "No contract address provided. Blockchain features disabled."
            )

    def is_connected(self) -> bool:
        """Check if connected to blockchain"""
        try:
            return self.w3.is_connected()
        except Exception as e:
            logger.error(f"Blockchain connection error: {e}")
            return False

    def record_flagged_prescription(
        self, prescription: PrescriptionData
    ) -> Optional[int]:
        """
        Record a flagged prescription on the blockchain

        Args:
            prescription: Prescription data to record

        Returns:
            Prescription ID if successful, None otherwise
        """
        if not self.contract:
            logger.error("Contract not initialized")
            return None

        if not self.is_connected():
            logger.error("Not connected to blockchain")
            return None

        try:
            # Convert risk factors to JSON string
            risk_factors_json = json.dumps(prescription.risk_factors)

            # Prepare transaction
            transaction = self.contract.functions.recordFlaggedPrescription(
                prescription.patient_id,
                prescription.doctor_id,
                prescription.pharmacy_id,
                prescription.drug_name,
                prescription.quantity,
                prescription.risk_score,
                risk_factors_json,
            ).build_transaction(
                {
                    "from": self.account,
                    "gas": 2000000,
                    "gasPrice": self.w3.eth.gas_price,
                    "nonce": self.w3.eth.get_transaction_count(self.account),
                }
            )

            # Send transaction using default account
            tx_hash = self.w3.eth.send_transaction(transaction)

            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                # Extract prescription ID from event logs
                prescription_id = self._extract_prescription_id_from_receipt(receipt)
                logger.info(f"Prescription {prescription_id} recorded on blockchain")
                return prescription_id
            else:
                logger.error("Transaction failed")
                return None

        except Exception as e:
            logger.error(f"Error recording prescription: {e}")
            return None

    def override_prescription(self, override_data: OverrideData) -> bool:
        """
        Override a flagged prescription

        Args:
            override_data: Override information

        Returns:
            True if successful, False otherwise
        """
        if not self.contract:
            logger.error("Contract not initialized")
            return False

        try:
            transaction = self.contract.functions.overridePrescription(
                override_data.prescription_id,
                override_data.override_reason,
                override_data.override_by,
            ).build_transaction(
                {
                    "from": self.account,
                    "gas": 1500000,
                    "gasPrice": self.w3.eth.gas_price,
                    "nonce": self.w3.eth.get_transaction_count(self.account),
                }
            )

            tx_hash = self.w3.eth.send_transaction(transaction)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                logger.info(f"Prescription {override_data.prescription_id} overridden")
                return True
            else:
                logger.error("Override transaction failed")
                return False

        except Exception as e:
            logger.error(f"Error overriding prescription: {e}")
            return False

    def verify_prescription(self, prescription_id: int, verified_by: str) -> bool:
        """
        Verify a prescription (for pharmacy use)

        Args:
            prescription_id: ID of prescription to verify
            verified_by: Identifier of person verifying

        Returns:
            True if successful, False otherwise
        """
        if not self.contract:
            logger.error("Contract not initialized")
            return False

        try:
            transaction = self.contract.functions.verifyPrescription(
                prescription_id, verified_by
            ).build_transaction(
                {
                    "from": self.account,
                    "gas": 1000000,
                    "gasPrice": self.w3.eth.gas_price,
                    "nonce": self.w3.eth.get_transaction_count(self.account),
                }
            )

            tx_hash = self.w3.eth.send_transaction(transaction)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                logger.info(f"Prescription {prescription_id} verified by {verified_by}")
                return True
            else:
                logger.error("Verification transaction failed")
                return False

        except Exception as e:
            logger.error(f"Error verifying prescription: {e}")
            return False

    def get_prescription_details(self, prescription_id: int) -> Optional[Dict]:
        """
        Get prescription details from blockchain

        Args:
            prescription_id: ID of prescription

        Returns:
            Prescription details dictionary or None
        """
        if not self.contract:
            logger.error("Contract not initialized")
            return None

        try:
            result = self.contract.functions.getPrescription(prescription_id).call()

            return {
                "id": result[0],
                "patient_id": result[1],
                "doctor_id": result[2],
                "pharmacy_id": result[3],
                "drug_name": result[4],
                "quantity": result[5],
                "timestamp": result[6],
                "risk_score": result[7],
                "risk_factors": json.loads(result[8]) if result[8] else {},
                "is_flagged": result[9],
                "is_overridden": result[10],
                "override_reason": result[11],
                "override_by": result[12],
                "override_timestamp": result[13],
            }

        except Exception as e:
            logger.error(f"Error getting prescription details: {e}")
            return None

    def is_prescription_flagged(self, prescription_id: int) -> Optional[bool]:
        """
        Check if prescription is currently flagged

        Args:
            prescription_id: ID of prescription

        Returns:
            True if flagged, False if not, None if error
        """
        if not self.contract:
            logger.error("Contract not initialized")
            return None

        try:
            return self.contract.functions.isPrescriptionFlagged(prescription_id).call()
        except Exception as e:
            logger.error(f"Error checking prescription flag status: {e}")
            return None

    def _extract_prescription_id_from_receipt(self, receipt) -> Optional[int]:
        """
        Extract prescription ID from transaction receipt events
        Looks for PrescriptionFlagged event and extracts the prescriptionId
        """
        try:
            # Get event signature
            event_signature = self.contract.events.PrescriptionFlagged()
            
            # Process logs to find matching event
            for log in receipt.logs:
                try:
                    # Try to decode this log
                    event_data = event_signature.process_log(log)
                    if event_data:
                        # Return the prescriptionId from the event
                        return event_data.args.prescriptionId
                except:
                    continue
            
            logger.error("PrescriptionFlagged event not found in receipt")
            return None

        except Exception as e:
            logger.error(f"Error extracting prescription ID: {e}")
            return None


class PrescriptionAuditManager:
    """
    High-level manager for prescription auditing operations
    Integrates AI models with blockchain recording
    """

    def __init__(self, blockchain_audit: BlockchainPrescriptionAudit):
        self.blockchain = blockchain_audit
        self.local_records = {}  # Local cache for offline operations

    def process_prescription(
        self,
        prescription_data: PrescriptionData,
        ai_risk_score: int,
        ai_risk_factors: Dict[str, any],
    ) -> Dict[str, any]:
        """
        Process a prescription through AI analysis and blockchain recording

        Args:
            prescription_data: Prescription information
            ai_risk_score: AI-calculated risk score
            ai_risk_factors: AI-identified risk factors

        Returns:
            Processing result dictionary
        """
        result = {
            "success": False,
            "prescription_id": None,
            "is_flagged": False,
            "risk_score": ai_risk_score,
            "risk_factors": ai_risk_factors,
            "blockchain_recorded": False,
            "error": None,
        }

        try:
            # Update prescription data with AI results
            prescription_data.risk_score = ai_risk_score
            prescription_data.risk_factors = ai_risk_factors
            prescription_data.timestamp = int(time.time())

            # Check if prescription should be flagged (risk score >= 70)
            # Load risk threshold from test data
            with open("test_data.json", "r") as f:
                test_data = json.load(f)
            flag_threshold = test_data["risk_thresholds"]["flag_threshold"]
            
            if ai_risk_score >= flag_threshold:
                result["is_flagged"] = True

                # Record on blockchain
                prescription_id = self.blockchain.record_flagged_prescription(
                    prescription_data
                )

                if prescription_id:
                    result["prescription_id"] = prescription_id
                    result["blockchain_recorded"] = True
                    result["success"] = True

                    # Store in local cache
                    self.local_records[prescription_id] = {
                        "prescription": prescription_data,
                        "timestamp": time.time(),
                        "status": "flagged",
                    }
                else:
                    result["error"] = "Failed to record on blockchain"
            else:
                result["success"] = True
                logger.info(f"Prescription not flagged (risk score: {ai_risk_score})")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing prescription: {e}")

        return result

    def override_prescription(
        self, prescription_id: int, override_reason: str, override_by: str
    ) -> bool:
        """
        Override a flagged prescription

        Args:
            prescription_id: ID of prescription to override
            override_reason: Reason for override
            override_by: Person overriding

        Returns:
            True if successful, False otherwise
        """
        try:
            override_data = OverrideData(
                prescription_id=prescription_id,
                override_reason=override_reason,
                override_by=override_by,
                timestamp=int(time.time()),
            )

            success = self.blockchain.override_prescription(override_data)

            if success and prescription_id in self.local_records:
                self.local_records[prescription_id]["status"] = "overridden"
                self.local_records[prescription_id]["override_reason"] = override_reason
                self.local_records[prescription_id]["override_by"] = override_by

            return success

        except Exception as e:
            logger.error(f"Error overriding prescription: {e}")
            return False

    def verify_prescription_for_pharmacy(
        self, prescription_id: int, verified_by: str
    ) -> bool:
        """
        Verify prescription for pharmacy dispensing

        Args:
            prescription_id: ID of prescription to verify
            verified_by: Pharmacist identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if prescription is flagged and not overridden
            is_flagged = self.blockchain.is_prescription_flagged(prescription_id)

            if is_flagged:
                logger.warning(
                    f"Prescription {prescription_id} is flagged - verification blocked"
                )
                return False

            # Record verification on blockchain
            success = self.blockchain.verify_prescription(prescription_id, verified_by)

            if success and prescription_id in self.local_records:
                self.local_records[prescription_id]["status"] = "verified"
                self.local_records[prescription_id]["verified_by"] = verified_by

            return success

        except Exception as e:
            logger.error(f"Error verifying prescription: {e}")
            return False

    def get_prescription_status(self, prescription_id: int) -> Optional[Dict]:
        """
        Get comprehensive prescription status

        Args:
            prescription_id: ID of prescription

        Returns:
            Status dictionary or None
        """
        try:
            # Get blockchain data
            blockchain_data = self.blockchain.get_prescription_details(prescription_id)

            # Get local data
            local_data = self.local_records.get(prescription_id, {})

            if blockchain_data:
                return {
                    "prescription_id": prescription_id,
                    "blockchain_data": blockchain_data,
                    "local_data": local_data,
                    "is_flagged": blockchain_data["is_flagged"],
                    "is_overridden": blockchain_data["is_overridden"],
                    "can_dispense": not blockchain_data["is_flagged"]
                    or blockchain_data["is_overridden"],
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting prescription status: {e}")
            return None


# Example usage and testing functions
def calculate_risk_score(features: Dict[str, any]) -> Tuple[int, Dict[str, any]]:
    """Calculate risk score based on patient features"""
    risk_score = 0
    risk_factors = {}
    
    # Previous opioid use
    if features["any_opioid_flag"] == 1:
        risk_score += 30
        risk_factors["previous_opioid_use"] = True
    
    # Multiple hospital admissions
    if features["n_hospital_admits"] > 1:
        risk_score += 15
        risk_factors["multiple_admissions"] = True
    
    # Long opioid exposure
    if features["opioid_exposure_days"] > 5:
        risk_score += 20
        risk_factors["extended_opioid_use"] = True
    
    # Benzodiazepine use
    if features["any_benzo_flag"] == 1:
        risk_score += 20
        risk_factors["benzo_use"] = True
    
    # Age risk (simplified)
    if features["age_at_first_admit"] > 65:
        risk_score += 15
        risk_factors["age_risk"] = "high"
    elif features["age_at_first_admit"] > 50:
        risk_score += 10
        risk_factors["age_risk"] = "moderate"
    
    return risk_score, risk_factors

def create_test_prescription() -> PrescriptionData:
    """Create a test prescription for demonstration"""
    # Load test data
    patient_data = load_patient_data("test_data.json")
    
    # Create handler and calculate features
    handler = PatientDataHandler()
    features = handler.convert_to_model_features({
        "patient_id": patient_data.patient_id,
        "demographics": {
            "age": patient_data.features["age_at_first_admit"],
            "gender": patient_data.features["gender"],
            "race": patient_data.features["race"],
            "insurance": patient_data.features["insurance"]
        },
        "medical_history": {
            "admissions": [
                {
                    "length_of_stay": patient_data.features["avg_los_days"]
                }
            ] * patient_data.features["n_hospital_admits"],
            "prescriptions": []  # We'll use the summary stats instead
        }
    })
    
    # Update features with medical history from test data
    features.update(patient_data.features)
    
    # Calculate risk score based on features
    risk_score, risk_factors = calculate_risk_score(features)
    
    return PrescriptionData(
        patient_id=patient_data.patient_id,
        doctor_id=patient_data.doctor_id,
        pharmacy_id=patient_data.pharmacy_id,
        drug_name=patient_data.drug_name,
        quantity=patient_data.quantity,
        risk_score=risk_score,
        risk_factors=risk_factors,
    )


def demo_blockchain_integration():
    """Demonstrate blockchain integration functionality"""
    print("=== AutoRxAudit Blockchain Integration Demo ===")

    # Initialize blockchain audit with config file
    try:
        blockchain_audit = BlockchainPrescriptionAudit(
            config_file="blockchain_config.json"
        )
        print("Successfully connected to blockchain network")
        print(f"Using contract at: {blockchain_audit.contract_address}")
        
        if not blockchain_audit.is_connected():
            print("Failed to connect to blockchain network")
            return
    except Exception as e:
        print(f"Failed to initialize blockchain connection: {e}")
        return

    # Initialize audit manager
    audit_manager = PrescriptionAuditManager(blockchain_audit)

    # Create test prescription
    prescription = create_test_prescription()

    print(f"\nProcessing prescription:")
    print(f"Drug: {prescription.drug_name}")
    print(f"Quantity: {prescription.quantity}")
    print(f"Patient ID: {prescription.patient_id}")
    print(f"Doctor ID: {prescription.doctor_id}")
    print(f"Risk Score: {prescription.risk_score}")
    print(f"Risk Factors: {json.dumps(prescription.risk_factors, indent=2)}")

    # Process prescription
    result = audit_manager.process_prescription(
        prescription, prescription.risk_score, prescription.risk_factors
    )

    print(f"\nProcessing Result:")
    print(f"Success: {result['success']}")
    print(f"Flagged: {result['is_flagged']}")
    print(f"Blockchain Recorded: {result['blockchain_recorded']}")

    if result["prescription_id"]:
        prescription_id = result["prescription_id"]
        print(f"Prescription ID: {prescription_id}")

        # Get prescription details from blockchain
        details = blockchain_audit.get_prescription_details(prescription_id)
        print("\nPrescription Details from Blockchain:")
        print(json.dumps(details, indent=2))

        # Demonstrate override
        print(f"\nDemonstrating prescription override...")
        # Load override data from test data
        with open("test_data.json", "r") as f:
            test_data = json.load(f)
        override_data = test_data["override"]
        
        override_success = audit_manager.override_prescription(
            prescription_id,
            override_data["reason"],
            override_data["by"]
        )
        print(f"Override Success: {override_success}")

        # Get updated details
        details = blockchain_audit.get_prescription_details(prescription_id)
        print("\nUpdated Prescription Details:")
        print(json.dumps(details, indent=2))

        # Demonstrate pharmacy verification
        print(f"\nDemonstrating pharmacy verification...")
        # Load verification data from test data
        with open("test_data.json", "r") as f:
            test_data = json.load(f)
        verification_data = test_data["verification"]
        
        verify_success = audit_manager.verify_prescription_for_pharmacy(
            prescription_id, verification_data["pharmacist_id"]
        )
        print(f"Verification Success: {verify_success}")
        
        # Check final flag status
        is_flagged = blockchain_audit.is_prescription_flagged(prescription_id)
        print(f"\nFinal Flag Status: {'Flagged' if is_flagged else 'Not Flagged'}")


if __name__ == "__main__":
    demo_blockchain_integration()
