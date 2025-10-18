"""
AI-Blockchain Integration Module for AutoRxAudit
Connects AI risk assessment with blockchain prescription auditing
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from blockchain_audit import (
    BlockchainPrescriptionAudit,
    PrescriptionAuditManager,
    PrescriptionData,
)
from pharmacy_verification import (
    PharmacyVerificationSystem,
    PrescriptionVerificationRequest,
)
from patient_data_handler import PatientDataHandler
from pycaret_model_wrapper import PyCaretModelWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Complete audit result combining AI and blockchain"""

    prescription_id: Optional[int]
    ai_result: AIPredictionResult
    blockchain_recorded: bool
    is_flagged: bool
    verification_status: str
    audit_timestamp: int
    error: Optional[str] = None


class AIBlockchainIntegrator:
    """
    Integrates AI models with blockchain prescription auditing
    """

    def __init__(
        self,
        audit_manager: PrescriptionAuditManager,
        ai_model_path: str = None,
        data_handler: Optional[PatientDataHandler] = None,
    ):
        self.audit_manager = audit_manager
        self.ai_model_path = ai_model_path
        self.ai_model = None
        self.data_handler = data_handler or PatientDataHandler()
        self.load_ai_model()

    def load_ai_model(self):
        """Load the trained AI model using PyCaret wrapper"""
        try:
            if self.ai_model_path:
                self.ai_model = PyCaretModelWrapper(self.ai_model_path)
                logger.info(f"AI model loaded from {self.ai_model_path}")
            else:
                logger.warning("No AI model path provided - using mock predictions")
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
            self.ai_model = None

    def predict_opioid_risk(self, patient_data: Dict[str, Any]) -> AIPredictionResult:
        """
        Predict opioid risk using AI model

        Args:
            patient_data: Dictionary of patient data in JSON format

        Returns:
            AI prediction result
        """
        try:
            # Validate patient data structure
            if not isinstance(patient_data, dict) or "patient_id" not in patient_data:
                raise ValueError(
                    "Invalid patient data format - missing required fields"
                )

            try:
                # Process patient data into features using data handler
                features = self.data_handler.convert_to_model_features(patient_data)
            except Exception as e:
                logger.error(f"Error converting patient data to features: {str(e)}")
                raise ValueError(f"Failed to process patient data: {str(e)}")

            if self.ai_model:
                try:
                    # Get prediction using model wrapper
                    prediction_result = self.ai_model.predict_risk(features)
                except Exception as e:
                    logger.error(f"Model prediction error: {str(e)}")
                    raise RuntimeError(f"Failed to generate prediction: {str(e)}")

                # Validate prediction result
                required_fields = ["risk_score", "confidence", "model_info"]
                if not all(field in prediction_result for field in required_fields):
                    raise ValueError("Invalid prediction result format from model")

                # Extract risk factors
                try:
                    risk_factors = self._extract_risk_factors(
                        features, prediction_result["risk_score"]
                    )
                except Exception as e:
                    logger.error(f"Error extracting risk factors: {str(e)}")
                    risk_factors = {"error": str(e)}

                return AIPredictionResult(
                    risk_score=prediction_result["risk_score"],
                    risk_factors=risk_factors,
                    confidence=prediction_result["confidence"],
                    model_used=prediction_result["model_info"],
                    prediction_timestamp=int(time.time()),
                )
            else:
                logger.warning("No AI model available - using mock prediction")
                return self._mock_prediction(features)

        except Exception as e:
            logger.error(f"Error in AI prediction: {e}")
            return AIPredictionResult(
                risk_score=50,
                risk_factors={"error": str(e)},
                confidence=0.5,
                model_used="Error_Fallback",
                prediction_timestamp=int(time.time()),
            )

    def _extract_risk_factors(
        self, features: Dict[str, Any], risk_score: int
    ) -> Dict[str, Any]:
        """Extract risk factors from prescription features"""
        risk_factors = {}

        # Check for high-risk indicators
        if features.get("age_at_first_admit", 0) < 25:
            risk_factors["young_age"] = True

        if features.get("opioid_rx_count", 0) > 0:
            risk_factors["previous_opioid_use"] = True

        if features.get("any_benzo_flag", 0) == 1:
            risk_factors["benzodiazepine_concurrent"] = True

        if features.get("n_hospital_admits", 0) > 3:
            risk_factors["frequent_hospitalizations"] = True

        if features.get("total_los_days", 0) > 30:
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

        return risk_factors

    def _mock_prediction(self, features: Dict[str, Any]) -> AIPredictionResult:
        """Generate mock prediction for demo purposes"""
        # Simple heuristic-based risk scoring
        risk_score = 30  # Base risk

        if features.get("opioid_rx_count", 0) > 0:
            risk_score += 30

        if features.get("any_benzo_flag", 0) == 1:
            risk_score += 20

        if features.get("age_at_first_admit", 0) < 25:
            risk_score += 15

        if features.get("n_hospital_admits", 0) > 2:
            risk_score += 10

        risk_score = min(risk_score, 100)

        risk_factors = self._extract_risk_factors(features, risk_score)

        return AIPredictionResult(
            risk_score=risk_score,
            risk_factors=risk_factors,
            confidence=0.85,
            model_used="Mock_Heuristic",
            prediction_timestamp=int(time.time()),
        )

    def audit_prescription(
        self, prescription_data: PrescriptionData, prescription_features: Dict[str, Any]
    ) -> PrescriptionAuditResult:
        """
        Complete prescription audit using AI and blockchain

        Args:
            prescription_data: Prescription information
            prescription_features: Features for AI prediction

        Returns:
            Complete audit result
        """
        audit_timestamp = int(time.time())

        try:
            # Get AI prediction
            ai_result = self.predict_opioid_risk(prescription_features)

            # Update prescription data with AI results
            prescription_data.risk_score = ai_result.risk_score
            prescription_data.risk_factors = ai_result.risk_factors
            prescription_data.timestamp = audit_timestamp

            # Process through blockchain audit manager
            result = self.audit_manager.process_prescription(
                prescription_data, ai_result.risk_score, ai_result.risk_factors
            )

            # Determine verification status
            if result["is_flagged"]:
                if result["blockchain_recorded"]:
                    verification_status = "flagged_recorded"
                else:
                    verification_status = "flagged_not_recorded"
            else:
                verification_status = "approved"

            return PrescriptionAuditResult(
                prescription_id=result.get("prescription_id"),
                ai_result=ai_result,
                blockchain_recorded=result["blockchain_recorded"],
                is_flagged=result["is_flagged"],
                verification_status=verification_status,
                audit_timestamp=audit_timestamp,
                error=result.get("error"),
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
                blockchain_recorded=False,
                is_flagged=False,
                verification_status="error",
                audit_timestamp=audit_timestamp,
                error=str(e),
            )


class PrescriptionAuditAPI:
    """
    High-level API for prescription auditing operations
    """

    def __init__(
        self,
        integrator: AIBlockchainIntegrator,
        verification_system: PharmacyVerificationSystem,
    ):
        self.integrator = integrator
        self.verification_system = verification_system
        self.audit_history = []

    def process_new_prescription(
        self,
        patient_data: Dict[str, Any],
        doctor_id: str,
        pharmacy_id: str,
        drug_name: str,
        quantity: int,
    ) -> PrescriptionAuditResult:
        """
        Process a new prescription through the complete audit system

        Args:
            patient_data: Full patient data in JSON format
            doctor_id: Doctor identifier
            pharmacy_id: Pharmacy identifier
            drug_name: Name of prescribed drug
            quantity: Quantity prescribed

        Returns:
            Complete audit result
        """
        audit_timestamp = int(time.time())

        try:
            # Validate input parameters
            if not patient_data or not isinstance(patient_data, dict):
                raise ValueError("Invalid patient data format")
            if not doctor_id or not isinstance(doctor_id, str):
                raise ValueError("Invalid doctor_id")
            if not pharmacy_id or not isinstance(pharmacy_id, str):
                raise ValueError("Invalid pharmacy_id")
            if not drug_name or not isinstance(drug_name, str):
                raise ValueError("Invalid drug_name")
            if not isinstance(quantity, int) or quantity <= 0:
                raise ValueError("Invalid quantity")

            # Get patient ID from data
            patient_id = patient_data.get("patient_id")
            if not patient_id:
                raise ValueError("Missing patient_id in patient data")

            # Create prescription data
            prescription_data = PrescriptionData(
                patient_id=patient_id,
                doctor_id=doctor_id,
                pharmacy_id=pharmacy_id,
                drug_name=drug_name,
                quantity=quantity,
                risk_score=0,  # Will be updated by AI
                risk_factors={},  # Will be updated by AI
                timestamp=audit_timestamp,
            )
        except ValueError as ve:
            logger.error(f"Validation error in process_new_prescription: {str(ve)}")
            return PrescriptionAuditResult(
                prescription_id=None,
                ai_result=AIPredictionResult(
                    risk_score=0,
                    risk_factors={},
                    confidence=0.0,
                    model_used="Error",
                    prediction_timestamp=audit_timestamp,
                ),
                blockchain_recorded=False,
                is_flagged=False,
                verification_status="error",
                audit_timestamp=audit_timestamp,
                error=f"Validation error: {str(ve)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error in process_new_prescription: {str(e)}")
            return PrescriptionAuditResult(
                prescription_id=None,
                ai_result=AIPredictionResult(
                    risk_score=0,
                    risk_factors={},
                    confidence=0.0,
                    model_used="Error",
                    prediction_timestamp=audit_timestamp,
                ),
                blockchain_recorded=False,
                is_flagged=False,
                verification_status="error",
                audit_timestamp=audit_timestamp,
                error=f"System error: {str(e)}",
            )

        # Perform audit
        result = self.integrator.audit_prescription(prescription_data, patient_data)

        # Log audit
        self._log_audit(result, prescription_data)

        return result

    def verify_prescription_for_dispensing(
        self, prescription_id: int, pharmacist_id: str, pharmacy_id: str
    ) -> Dict[str, Any]:
        """
        Verify prescription for pharmacy dispensing

        Args:
            prescription_id: ID of prescription to verify
            pharmacist_id: Pharmacist identifier
            pharmacy_id: Pharmacy identifier

        Returns:
            Verification result
        """
        try:
            # Get prescription details
            details = self.verification_system.get_prescription_details(prescription_id)

            if not details:
                return {
                    "success": False,
                    "message": "Prescription not found",
                    "can_dispense": False,
                }

            # Create verification request
            blockchain_data = details["blockchain_data"]
            request = PrescriptionVerificationRequest(
                prescription_id=prescription_id,
                patient_id=blockchain_data["patient_id"],
                doctor_id=blockchain_data["doctor_id"],
                pharmacy_id=pharmacy_id,
                drug_name=blockchain_data["drug_name"],
                quantity=blockchain_data["quantity"],
                requested_by=pharmacist_id,
            )

            # Verify prescription
            verification_result = self.verification_system.verify_prescription(request)

            return {
                "success": True,
                "can_dispense": verification_result.is_valid,
                "verification_status": verification_result.verification_status,
                "message": verification_result.message,
                "is_flagged": verification_result.is_flagged,
                "risk_score": verification_result.risk_score,
                "verification_result": verification_result,
            }

        except Exception as e:
            logger.error(f"Error verifying prescription for dispensing: {e}")
            return {
                "success": False,
                "message": f"Verification error: {str(e)}",
                "can_dispense": False,
            }

    def request_prescription_override(
        self, prescription_id: int, override_reason: str, requested_by: str
    ) -> Dict[str, Any]:
        """
        Request override for a flagged prescription

        Args:
            prescription_id: ID of prescription to override
            override_reason: Reason for override
            requested_by: Person requesting override

        Returns:
            Override request result
        """
        try:
            success = self.verification_system.request_override(
                prescription_id, override_reason, requested_by
            )

            return {
                "success": success,
                "message": (
                    "Override request processed"
                    if success
                    else "Override request failed"
                ),
                "prescription_id": prescription_id,
            }

        except Exception as e:
            logger.error(f"Error requesting override: {e}")
            return {
                "success": False,
                "message": f"Override error: {str(e)}",
                "prescription_id": prescription_id,
            }

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit activities"""
        total_audits = len(self.audit_history)
        flagged_prescriptions = sum(
            1 for audit in self.audit_history if audit["is_flagged"]
        )
        blockchain_recorded = sum(
            1 for audit in self.audit_history if audit["blockchain_recorded"]
        )

        return {
            "total_prescriptions_audited": total_audits,
            "flagged_prescriptions": flagged_prescriptions,
            "blockchain_recorded": blockchain_recorded,
            "flag_rate": (
                (flagged_prescriptions / total_audits * 100) if total_audits > 0 else 0
            ),
            "audit_history": self.audit_history[-10:],  # Last 10 audits
        }

    def _log_audit(
        self, result: PrescriptionAuditResult, prescription_data: PrescriptionData
    ):
        """Log audit result"""
        log_entry = {
            "timestamp": result.audit_timestamp,
            "prescription_id": result.prescription_id,
            "patient_id": prescription_data.patient_id,
            "doctor_id": prescription_data.doctor_id,
            "drug_name": prescription_data.drug_name,
            "risk_score": result.ai_result.risk_score,
            "is_flagged": result.is_flagged,
            "blockchain_recorded": result.blockchain_recorded,
            "verification_status": result.verification_status,
            "model_used": result.ai_result.model_used,
        }
        self.audit_history.append(log_entry)


def demo_ai_blockchain_integration():
    """Demonstrate AI-Blockchain integration"""
    print("=== AutoRxAudit AI-Blockchain Integration Demo ===")

    # Initialize components
    from blockchain_audit import BlockchainPrescriptionAudit, PrescriptionAuditManager
    from pharmacy_verification import PharmacyVerificationSystem

    blockchain_audit = BlockchainPrescriptionAudit(
        rpc_url="http://localhost:8546",  # Using our Ganache instance
        contract_address="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",  # First test account from Ganache
        private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # Private key from first test account
    )

    audit_manager = PrescriptionAuditManager(blockchain_audit)
    verification_system = PharmacyVerificationSystem(audit_manager)

    # Initialize AI-Blockchain integrator with trained model
    model_path = "/Users/rukshik/Documents/GitHub/AutoRxAudit/best_model_shap_will_get_opioid_rx.pkl"  # Using SHAP-based model
    integrator = AIBlockchainIntegrator(audit_manager, ai_model_path=model_path)

    # Initialize API
    api = PrescriptionAuditAPI(integrator, verification_system)

    # Load demo patient data from JSON
    patient_data = {
        "patient_id": "PATIENT_001",
        "demographics": {
            "age": 35,
            "gender": "M",
            "race": "White",
            "insurance": "Medicare",
        },
        "medical_history": {
            "admissions": [
                {
                    "admission_id": "ADM001",
                    "admission_date": "2023-01-15",
                    "discharge_date": "2023-01-20",
                    "length_of_stay": 5,
                },
                {
                    "admission_id": "ADM002",
                    "admission_date": "2023-03-10",
                    "discharge_date": "2023-03-16",
                    "length_of_stay": 6,
                },
            ],
            "prescriptions": [
                {
                    "drug_name": "Hydrocodone",
                    "drug_class": "N02A",
                    "start_date": "2023-01-18",
                    "end_date": "2023-01-25",
                    "is_opioid": True,
                },
                {
                    "drug_name": "Lisinopril",
                    "drug_class": "C09A",
                    "start_date": "2023-01-18",
                    "is_opioid": False,
                },
            ],
        },
    }

    print("\n1. Processing new prescription...")
    result = api.process_new_prescription(
        patient_data=patient_data,
        doctor_id="DR_SMITH",
        pharmacy_id="PHARMACY_ABC",
        drug_name="Oxycodone",
        quantity=30,
    )

    print(f"Audit Result:")
    print(f"  Prescription ID: {result.prescription_id}")
    print(f"  Risk Score: {result.ai_result.risk_score}/100")
    print(f"  Confidence: {result.ai_result.confidence:.2f}")
    print(f"  Model Used: {result.ai_result.model_used}")
    print(f"  Flagged: {result.is_flagged}")
    print(f"  Blockchain Recorded: {result.blockchain_recorded}")
    print(f"  Risk Factors: {result.ai_result.risk_factors}")

    if result.prescription_id:
        print(f"\n2. Verifying prescription for dispensing...")
        verification_result = api.verify_prescription_for_dispensing(
            prescription_id=result.prescription_id,
            pharmacist_id="PHARMACIST_JONES",
            pharmacy_id="PHARMACY_ABC",
        )

        print(f"Verification Result:")
        print(f"  Can Dispense: {verification_result['can_dispense']}")
        print(f"  Status: {verification_result['verification_status']}")
        print(f"  Message: {verification_result['message']}")

        if not verification_result["can_dispense"]:
            print(f"\n3. Requesting override...")
            override_result = api.request_prescription_override(
                prescription_id=result.prescription_id,
                override_reason="Patient has documented chronic pain condition",
                requested_by="DR_SMITH",
            )

            print(f"Override Result: {override_result['message']}")

    print(f"\n4. Audit Summary:")
    summary = api.get_audit_summary()
    print(f"  Total Audits: {summary['total_prescriptions_audited']}")
    print(f"  Flagged: {summary['flagged_prescriptions']}")
    print(f"  Flag Rate: {summary['flag_rate']:.1f}%")


if __name__ == "__main__":
    demo_ai_blockchain_integration()
