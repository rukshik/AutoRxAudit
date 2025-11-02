"""
Simplified AutoRxAudit Demo (No Blockchain Dependencies)
This version demonstrates the AI functionality without blockchain components
"""

import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

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
    risk_factors: Dict[str, Any]
    timestamp: Optional[int] = None


@dataclass
class AIPredictionResult:
    """Result from AI model prediction"""

    risk_score: int
    risk_factors: Dict[str, Any]
    confidence: float
    model_used: str
    prediction_timestamp: int


@dataclass
class PrescriptionAuditResult:
    """Complete audit result"""

    prescription_id: Optional[int]
    ai_result: AIPredictionResult
    is_flagged: bool
    audit_timestamp: int
    error: Optional[str] = None


class SimplePrescriptionAudit:
    """
    Simplified prescription audit system without blockchain
    """

    def __init__(self):
        self.prescription_counter = 0
        self.audit_log = []
        self.flagged_prescriptions = {}

    def predict_opioid_risk(
        self, prescription_features: Dict[str, Any]
    ) -> AIPredictionResult:
        """
        Predict opioid risk using simple heuristics
        """
        # Simple heuristic-based risk scoring
        risk_score = 30  # Base risk

        if prescription_features.get("opioid_rx_count", 0) > 0:
            risk_score += 30

        if prescription_features.get("any_benzo_flag", 0) == 1:
            risk_score += 20

        if prescription_features.get("age_at_first_admit", 0) < 25:
            risk_score += 15

        if prescription_features.get("n_hospital_admits", 0) > 2:
            risk_score += 10

        risk_score = min(risk_score, 100)

        # Extract risk factors
        risk_factors = {}
        if prescription_features.get("age_at_first_admit", 0) < 25:
            risk_factors["young_age"] = True

        if prescription_features.get("opioid_rx_count", 0) > 0:
            risk_factors["previous_opioid_use"] = True

        if prescription_features.get("any_benzo_flag", 0) == 1:
            risk_factors["benzodiazepine_concurrent"] = True

        if prescription_features.get("n_hospital_admits", 0) > 3:
            risk_factors["frequent_hospitalizations"] = True

        if prescription_features.get("total_los_days", 0) > 30:
            risk_factors["extended_hospital_stay"] = True

        # Add risk score category
        if risk_score >= 80:
            risk_factors["risk_level"] = "very_high"
        elif risk_score >= 60:
            risk_factors["risk_level"] = "high"
        elif risk_score >= 40:
            risk_factors["risk_level"] = "moderate"
        else:
            risk_factors["risk_level"] = "low"

        return AIPredictionResult(
            risk_score=risk_score,
            risk_factors=risk_factors,
            confidence=0.85,
            model_used="Simple_Heuristic",
            prediction_timestamp=int(time.time()),
        )

    def audit_prescription(
        self, prescription_data: PrescriptionData, prescription_features: Dict[str, Any]
    ) -> PrescriptionAuditResult:
        """
        Complete prescription audit using AI
        """
        audit_timestamp = int(time.time())

        try:
            # Get AI prediction
            ai_result = self.predict_opioid_risk(prescription_features)

            # Update prescription data with AI results
            prescription_data.risk_score = ai_result.risk_score
            prescription_data.risk_factors = ai_result.risk_factors
            prescription_data.timestamp = audit_timestamp

            # Check if prescription should be flagged (risk score >= 70)
            is_flagged = ai_result.risk_score >= 70

            if is_flagged:
                self.prescription_counter += 1
                prescription_id = self.prescription_counter

                # Store flagged prescription
                self.flagged_prescriptions[prescription_id] = {
                    "prescription": prescription_data,
                    "ai_result": ai_result,
                    "timestamp": audit_timestamp,
                    "status": "flagged",
                }

                logger.info(
                    f"Prescription {prescription_id} flagged with risk score {ai_result.risk_score}"
                )
            else:
                prescription_id = None
                logger.info(
                    f"Prescription not flagged (risk score: {ai_result.risk_score})"
                )

            # Log audit
            self._log_audit(
                prescription_id,
                prescription_data,
                ai_result,
                is_flagged,
                audit_timestamp,
            )

            return PrescriptionAuditResult(
                prescription_id=prescription_id,
                ai_result=ai_result,
                is_flagged=is_flagged,
                audit_timestamp=audit_timestamp,
            )

        except Exception as e:
            logger.error(f"Error in prescription audit: {e}")
            return PrescriptionAuditResult(
                prescription_id=None,
                ai_result=AIPredictionResult(
                    risk_score=0,
                    risk_factors={},
                    confidence=0.0,
                    model_used="Error",
                    prediction_timestamp=audit_timestamp,
                ),
                is_flagged=False,
                audit_timestamp=audit_timestamp,
                error=str(e),
            )

    def override_prescription(
        self, prescription_id: int, override_reason: str, override_by: str
    ) -> bool:
        """
        Override a flagged prescription
        """
        try:
            if prescription_id in self.flagged_prescriptions:
                self.flagged_prescriptions[prescription_id]["status"] = "overridden"
                self.flagged_prescriptions[prescription_id][
                    "override_reason"
                ] = override_reason
                self.flagged_prescriptions[prescription_id]["override_by"] = override_by
                self.flagged_prescriptions[prescription_id]["override_timestamp"] = int(
                    time.time()
                )

                logger.info(
                    f"Prescription {prescription_id} overridden by {override_by}"
                )
                return True
            else:
                logger.error(f"Prescription {prescription_id} not found")
                return False

        except Exception as e:
            logger.error(f"Error overriding prescription: {e}")
            return False

    def verify_prescription(
        self, prescription_id: int, verified_by: str
    ) -> Dict[str, Any]:
        """
        Verify a prescription for dispensing
        """
        try:
            if prescription_id not in self.flagged_prescriptions:
                return {
                    "success": False,
                    "message": "Prescription not found",
                    "can_dispense": False,
                }

            prescription_data = self.flagged_prescriptions[prescription_id]

            # Check if prescription is flagged and not overridden
            if prescription_data["status"] == "flagged":
                return {
                    "success": True,
                    "can_dispense": False,
                    "verification_status": "blocked",
                    "message": f"Prescription blocked - flagged with risk score {prescription_data['ai_result'].risk_score}",
                    "is_flagged": True,
                    "risk_score": prescription_data["ai_result"].risk_score,
                }
            elif prescription_data["status"] == "overridden":
                return {
                    "success": True,
                    "can_dispense": True,
                    "verification_status": "approved",
                    "message": f"Prescription approved - overridden by {prescription_data['override_by']}",
                    "is_flagged": True,
                    "is_overridden": True,
                    "override_reason": prescription_data["override_reason"],
                }
            else:
                return {
                    "success": True,
                    "can_dispense": True,
                    "verification_status": "approved",
                    "message": "Prescription approved",
                    "is_flagged": False,
                }

        except Exception as e:
            logger.error(f"Error verifying prescription: {e}")
            return {
                "success": False,
                "message": f"Verification error: {str(e)}",
                "can_dispense": False,
            }

    def get_prescription_details(self, prescription_id: int) -> Optional[Dict]:
        """
        Get prescription details
        """
        if prescription_id in self.flagged_prescriptions:
            return self.flagged_prescriptions[prescription_id]
        return None

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit activities"""
        total_audits = len(self.audit_log)
        flagged_prescriptions = sum(
            1 for audit in self.audit_log if audit["is_flagged"]
        )

        return {
            "total_prescriptions_audited": total_audits,
            "flagged_prescriptions": flagged_prescriptions,
            "flag_rate": (
                (flagged_prescriptions / total_audits * 100) if total_audits > 0 else 0
            ),
            "audit_history": self.audit_log[-10:],  # Last 10 audits
        }

    def _log_audit(
        self,
        prescription_id: Optional[int],
        prescription_data: PrescriptionData,
        ai_result: AIPredictionResult,
        is_flagged: bool,
        timestamp: int,
    ):
        """Log audit result"""
        log_entry = {
            "timestamp": timestamp,
            "prescription_id": prescription_id,
            "patient_id": prescription_data.patient_id,
            "doctor_id": prescription_data.doctor_id,
            "drug_name": prescription_data.drug_name,
            "risk_score": ai_result.risk_score,
            "is_flagged": is_flagged,
            "model_used": ai_result.model_used,
        }
        self.audit_log.append(log_entry)


def demo_simple_audit():
    """Demonstrate simplified prescription audit system"""
    print("=== AutoRxAudit Simplified Demo (No Blockchain) ===")

    # Initialize audit system
    audit_system = SimplePrescriptionAudit()

    # Demo prescription features
    prescription_features = {
        "age_at_first_admit": 35,
        "gender": 1,  # Male
        "race": 2,  # White
        "insurance": 1,  # Medicare
        "n_hospital_admits": 2,
        "avg_los_days": 5.5,
        "total_los_days": 11.0,
        "opioid_rx_count": 1,
        "opioid_hadms": 1,
        "distinct_opioids": 1,
        "opioid_exposure_days": 7.0,
        "any_benzo_flag": 0,
        "any_opioid_flag": 1,
        "atc_A_rx_count": 2,
        "atc_C_rx_count": 1,
        "atc_N_rx_count": 1,
    }

    print("\n1. Processing new prescription...")

    # Create prescription data
    prescription_data = PrescriptionData(
        patient_id="PATIENT_001",
        doctor_id="DR_SMITH",
        pharmacy_id="PHARMACY_ABC",
        drug_name="Oxycodone",
        quantity=30,
        risk_score=0,  # Will be updated by AI
        risk_factors={},  # Will be updated by AI
        timestamp=int(time.time()),
    )

    # Process prescription
    result = audit_system.audit_prescription(prescription_data, prescription_features)

    print(f"Audit Result:")
    print(f"  Prescription ID: {result.prescription_id}")
    print(f"  Risk Score: {result.ai_result.risk_score}/100")
    print(f"  Confidence: {result.ai_result.confidence:.2f}")
    print(f"  Model Used: {result.ai_result.model_used}")
    print(f"  Flagged: {result.is_flagged}")
    print(f"  Risk Factors: {result.ai_result.risk_factors}")

    if result.prescription_id:
        print(f"\n2. Verifying prescription for dispensing...")
        verification_result = audit_system.verify_prescription(
            prescription_id=result.prescription_id, verified_by="PHARMACIST_JONES"
        )

        print(f"Verification Result:")
        print(f"  Can Dispense: {verification_result['can_dispense']}")
        print(f"  Status: {verification_result['verification_status']}")
        print(f"  Message: {verification_result['message']}")

        if not verification_result["can_dispense"]:
            print(f"\n3. Requesting override...")
            override_success = audit_system.override_prescription(
                prescription_id=result.prescription_id,
                override_reason="Patient has documented chronic pain condition",
                override_by="DR_SMITH",
            )

            print(f"Override Success: {override_success}")

            if override_success:
                print(f"\n4. Re-verifying after override...")
                verification_result = audit_system.verify_prescription(
                    prescription_id=result.prescription_id,
                    verified_by="PHARMACIST_JONES",
                )

                print(f"Verification Result:")
                print(f"  Can Dispense: {verification_result['can_dispense']}")
                print(f"  Status: {verification_result['verification_status']}")
                print(f"  Message: {verification_result['message']}")

    print(f"\n5. Audit Summary:")
    summary = audit_system.get_audit_summary()
    print(f"  Total Audits: {summary['total_prescriptions_audited']}")
    print(f"  Flagged: {summary['flagged_prescriptions']}")
    print(f"  Flag Rate: {summary['flag_rate']:.1f}%")

    print(f"\n6. Prescription Details:")
    if result.prescription_id:
        details = audit_system.get_prescription_details(result.prescription_id)
        if details:
            print(f"  Patient: {details['prescription'].patient_id}")
            print(f"  Doctor: {details['prescription'].doctor_id}")
            print(f"  Drug: {details['prescription'].drug_name}")
            print(f"  Status: {details['status']}")
            if details["status"] == "overridden":
                print(f"  Override Reason: {details['override_reason']}")
                print(f"  Overridden By: {details['override_by']}")


if __name__ == "__main__":
    demo_simple_audit()
