"""
Pharmacy Verification System for AutoRxAudit
Allows pharmacists to verify prescriptions against blockchain records
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from blockchain_audit import BlockchainPrescriptionAudit, PrescriptionAuditManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PrescriptionVerificationRequest:
    """Request for prescription verification"""

    prescription_id: int
    patient_id: str
    doctor_id: str
    pharmacy_id: str
    drug_name: str
    quantity: int
    requested_by: str  # Pharmacist ID
    timestamp: Optional[int] = None


@dataclass
class VerificationResult:
    """Result of prescription verification"""

    prescription_id: int
    is_valid: bool
    is_flagged: bool
    is_overridden: bool
    risk_score: Optional[int]
    risk_factors: Optional[Dict]
    override_reason: Optional[str]
    override_by: Optional[str]
    verification_status: str  # 'approved', 'blocked', 'requires_override'
    message: str
    timestamp: int


class PharmacyVerificationSystem:
    """
    Pharmacy verification system for checking prescriptions against blockchain
    """

    def __init__(self, audit_manager: PrescriptionAuditManager):
        self.audit_manager = audit_manager
        self.verification_log = []
        self.pharmacy_permissions = {}  # pharmacy_id -> permissions

    def verify_prescription(
        self, request: PrescriptionVerificationRequest
    ) -> VerificationResult:
        """
        Verify a prescription for pharmacy dispensing

        Args:
            request: Verification request details

        Returns:
            Verification result with approval status
        """
        timestamp = int(time.time())

        try:
            # Get prescription status from blockchain
            status = self.audit_manager.get_prescription_status(request.prescription_id)

            if not status:
                return VerificationResult(
                    prescription_id=request.prescription_id,
                    is_valid=False,
                    is_flagged=False,
                    is_overridden=False,
                    risk_score=None,
                    risk_factors=None,
                    override_reason=None,
                    override_by=None,
                    verification_status="blocked",
                    message="Prescription not found in blockchain records",
                    timestamp=timestamp,
                )

            blockchain_data = status["blockchain_data"]
            is_flagged = blockchain_data["is_flagged"]
            is_overridden = blockchain_data["is_overridden"]
            risk_score = blockchain_data["risk_score"]
            risk_factors = blockchain_data["risk_factors"]

            # Determine verification status
            if not is_flagged:
                verification_status = "approved"
                message = "Prescription approved - no flags detected"
            elif is_overridden:
                verification_status = "approved"
                message = f'Prescription approved - previously overridden by {blockchain_data["override_by"]}'
            else:
                verification_status = "blocked"
                message = f"Prescription blocked - flagged with risk score {risk_score}"

            # Create verification result
            result = VerificationResult(
                prescription_id=request.prescription_id,
                is_valid=(verification_status == "approved"),
                is_flagged=is_flagged,
                is_overridden=is_overridden,
                risk_score=risk_score,
                risk_factors=risk_factors,
                override_reason=blockchain_data.get("override_reason"),
                override_by=blockchain_data.get("override_by"),
                verification_status=verification_status,
                message=message,
                timestamp=timestamp,
            )

            # Log verification attempt
            self._log_verification(request, result)

            # Record verification on blockchain if approved
            if verification_status == "approved":
                self.audit_manager.verify_prescription_for_pharmacy(
                    request.prescription_id, request.requested_by
                )

            return result

        except Exception as e:
            logger.error(f"Error verifying prescription: {e}")
            return VerificationResult(
                prescription_id=request.prescription_id,
                is_valid=False,
                is_flagged=False,
                is_overridden=False,
                risk_score=None,
                risk_factors=None,
                override_reason=None,
                override_by=None,
                verification_status="blocked",
                message=f"Verification error: {str(e)}",
                timestamp=timestamp,
            )

    def request_override(
        self, prescription_id: int, override_reason: str, requested_by: str
    ) -> bool:
        """
        Request override for a flagged prescription

        Args:
            prescription_id: ID of prescription to override
            override_reason: Reason for override request
            requested_by: Person requesting override

        Returns:
            True if override request processed successfully
        """
        try:
            success = self.audit_manager.override_prescription(
                prescription_id, override_reason, requested_by
            )

            if success:
                logger.info(
                    f"Override request processed for prescription {prescription_id}"
                )
                self._log_override_request(
                    prescription_id, override_reason, requested_by
                )

            return success

        except Exception as e:
            logger.error(f"Error processing override request: {e}")
            return False

    def get_prescription_details(self, prescription_id: int) -> Optional[Dict]:
        """
        Get detailed prescription information

        Args:
            prescription_id: ID of prescription

        Returns:
            Prescription details dictionary or None
        """
        try:
            status = self.audit_manager.get_prescription_status(prescription_id)
            return status
        except Exception as e:
            logger.error(f"Error getting prescription details: {e}")
            return None

    def get_verification_history(self, pharmacy_id: str = None) -> List[Dict]:
        """
        Get verification history

        Args:
            pharmacy_id: Optional pharmacy ID to filter by

        Returns:
            List of verification records
        """
        if pharmacy_id:
            return [
                log
                for log in self.verification_log
                if log.get("pharmacy_id") == pharmacy_id
            ]
        return self.verification_log.copy()

    def _log_verification(
        self, request: PrescriptionVerificationRequest, result: VerificationResult
    ):
        """Log verification attempt"""
        log_entry = {
            "timestamp": result.timestamp,
            "prescription_id": request.prescription_id,
            "pharmacy_id": request.pharmacy_id,
            "requested_by": request.requested_by,
            "verification_status": result.verification_status,
            "is_flagged": result.is_flagged,
            "is_overridden": result.is_overridden,
            "risk_score": result.risk_score,
            "message": result.message,
        }
        self.verification_log.append(log_entry)

    def _log_override_request(
        self, prescription_id: int, reason: str, requested_by: str
    ):
        """Log override request"""
        log_entry = {
            "timestamp": int(time.time()),
            "prescription_id": prescription_id,
            "action": "override_request",
            "reason": reason,
            "requested_by": requested_by,
        }
        self.verification_log.append(log_entry)


class PharmacyDashboard:
    """
    Simple dashboard for pharmacists to interact with the verification system
    """

    def __init__(self, verification_system: PharmacyVerificationSystem):
        self.verification_system = verification_system

    def display_prescription_status(self, prescription_id: int):
        """Display prescription status in a user-friendly format"""
        details = self.verification_system.get_prescription_details(prescription_id)

        if not details:
            print(f"❌ Prescription {prescription_id} not found")
            return

        blockchain_data = details["blockchain_data"]

        print(f"\n📋 Prescription Details - ID: {prescription_id}")
        print(f"{'='*50}")
        print(f"Patient ID: {blockchain_data['patient_id']}")
        print(f"Doctor ID: {blockchain_data['doctor_id']}")
        print(f"Drug: {blockchain_data['drug_name']}")
        print(f"Quantity: {blockchain_data['quantity']}")
        print(f"Risk Score: {blockchain_data['risk_score']}/100")

        if blockchain_data["is_flagged"]:
            print(f"🚨 Status: FLAGGED")
            if blockchain_data["is_overridden"]:
                print(f"✅ Override: YES")
                print(f"   Reason: {blockchain_data['override_reason']}")
                print(f"   Overridden by: {blockchain_data['override_by']}")
            else:
                print(f"❌ Override: NO")
        else:
            print(f"✅ Status: CLEAR")

        print(f"Can Dispense: {'YES' if details['can_dispense'] else 'NO'}")

        # Display risk factors
        if blockchain_data["risk_factors"]:
            print(f"\nRisk Factors:")
            for factor, value in blockchain_data["risk_factors"].items():
                print(f"  • {factor}: {value}")

    def process_verification_request(
        self,
        prescription_id: int,
        patient_id: str,
        doctor_id: str,
        pharmacy_id: str,
        drug_name: str,
        quantity: int,
        pharmacist_id: str,
    ) -> VerificationResult:
        """Process a verification request and display results"""

        request = PrescriptionVerificationRequest(
            prescription_id=prescription_id,
            patient_id=patient_id,
            doctor_id=doctor_id,
            pharmacy_id=pharmacy_id,
            drug_name=drug_name,
            quantity=quantity,
            requested_by=pharmacist_id,
            timestamp=int(time.time()),
        )

        print(f"\n🔍 Verifying Prescription {prescription_id}...")
        print(f"Drug: {drug_name}")
        print(f"Quantity: {quantity}")
        print(f"Pharmacist: {pharmacist_id}")

        result = self.verification_system.verify_prescription(request)

        print(f"\n📊 Verification Result:")
        print(f"{'='*30}")

        if result.verification_status == "approved":
            print(f"✅ APPROVED - Prescription can be dispensed")
        else:
            print(f"❌ BLOCKED - Prescription cannot be dispensed")

        print(f"Status: {result.verification_status.upper()}")
        print(f"Message: {result.message}")

        if result.is_flagged:
            print(f"Risk Score: {result.risk_score}/100")
            if result.risk_factors:
                print(f"Risk Factors: {result.risk_factors}")

        return result

    def request_override(self, prescription_id: int, reason: str, pharmacist_id: str):
        """Request override for a flagged prescription"""
        print(f"\n🔄 Requesting Override for Prescription {prescription_id}...")
        print(f"Reason: {reason}")
        print(f"Requested by: {pharmacist_id}")

        success = self.verification_system.request_override(
            prescription_id, reason, pharmacist_id
        )

        if success:
            print(f"✅ Override request processed successfully")
        else:
            print(f"❌ Override request failed")

        return success


def demo_pharmacy_verification():
    """Demonstrate pharmacy verification system"""
    print("=== AutoRxAudit Pharmacy Verification Demo ===")

    # Initialize blockchain components (mock for demo)
    from blockchain_audit import BlockchainPrescriptionAudit, PrescriptionAuditManager

    blockchain_audit = BlockchainPrescriptionAudit(
        rpc_url="http://localhost:8545",
        contract_address="0x1234567890123456789012345678901234567890",
        private_key="0x1234567890123456789012345678901234567890123456789012345678901234",
    )

    audit_manager = PrescriptionAuditManager(blockchain_audit)
    verification_system = PharmacyVerificationSystem(audit_manager)
    dashboard = PharmacyDashboard(verification_system)

    # Demo prescription verification
    print("\n1. Verifying a flagged prescription...")
    result = dashboard.process_verification_request(
        prescription_id=1,
        patient_id="PATIENT_001",
        doctor_id="DR_SMITH",
        pharmacy_id="PHARMACY_ABC",
        drug_name="Oxycodone",
        quantity=30,
        pharmacist_id="PHARMACIST_JONES",
    )

    # Demo override request
    if result.verification_status == "blocked":
        print("\n2. Requesting override for blocked prescription...")
        dashboard.request_override(
            prescription_id=1,
            reason="Patient has documented chronic pain condition",
            pharmacist_id="PHARMACIST_JONES",
        )

        # Re-verify after override
        print("\n3. Re-verifying after override...")
        dashboard.process_verification_request(
            prescription_id=1,
            patient_id="PATIENT_001",
            doctor_id="DR_SMITH",
            pharmacy_id="PHARMACY_ABC",
            drug_name="Oxycodone",
            quantity=30,
            pharmacist_id="PHARMACIST_JONES",
        )

    # Display prescription details
    print("\n4. Displaying prescription details...")
    dashboard.display_prescription_status(1)

    # Show verification history
    print("\n5. Verification History:")
    history = verification_system.get_verification_history()
    for entry in history:
        print(
            f"  {entry['timestamp']}: Prescription {entry['prescription_id']} - {entry['verification_status']}"
        )


if __name__ == "__main__":
    demo_pharmacy_verification()
