"""
Patient Data Handler Module for AutoRxAudit
Loads and processes patient data from JSON files
"""

import json
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatientData:
    """Data structure for patient information"""

    patient_id: str
    doctor_id: str
    pharmacy_id: str
    drug_name: str
    quantity: int
    features: Dict[str, Any]


def load_patient_data(json_path: str) -> PatientData:
    """
    Load patient data from JSON file and convert to PatientData object

    Args:
        json_path: Path to the JSON file containing patient data

    Returns:
        PatientData object with prescription and feature information
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract prescription details
        prescription = data.get("prescription", {})

        # Combine all features for ML model
        features = {}

        # Add patient info features
        features.update(data.get("patient_info", {}))

        # Add medical history features
        features.update(data.get("medical_history", {}))

        # Remove non-feature fields that shouldn't go to the model
        feature_exclude = ["patient_id"]
        for field in feature_exclude:
            features.pop(field, None)

        return PatientData(
            patient_id=data["patient_info"]["patient_id"],
            doctor_id=prescription["doctor_id"],
            pharmacy_id=prescription["pharmacy_id"],
            drug_name=prescription["drug_name"],
            quantity=prescription["quantity"],
            features=features,
        )

    except Exception as e:
        raise ValueError(f"Error loading patient data: {str(e)}")


class PatientDataHandler:
    """Handler for loading and processing patient data"""

    def __init__(self):
        """Initialize the patient data handler"""
        self.required_fields = {
            "patient_id",
            "demographics",
            "medical_history",
        }

    def convert_to_model_features(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert patient data to format expected by ML model

        Args:
            patient_data: Dictionary containing patient data in JSON format

        Returns:
            Dictionary of features in format expected by model
        """
        try:
            # Validate required fields
            if not all(
                field in patient_data
                for field in ["patient_id", "demographics", "medical_history"]
            ):
                raise ValueError("Missing required fields in patient data")

            # Extract features
            features = {}

            # Demographics features
            demo = patient_data["demographics"]
            features["age_at_first_admit"] = demo.get("age", 0)
            features["gender"] = 1 if demo.get("gender") == "M" else 0
            features["race"] = demo.get("race", "Unknown")
            features["insurance"] = demo.get("insurance", "Unknown")

            # Medical history features
            med_hist = patient_data["medical_history"]

            # Process admissions
            admissions = med_hist.get("admissions", [])
            prescriptions = med_hist.get("prescriptions", [])
            features["n_hospital_admits"] = len(admissions)
            features["opioid_hadms"] = len([
                adm for adm in admissions 
                if any(rx.get("is_opioid", False) for rx in prescriptions 
                      if pd.to_datetime(rx.get("start_date", "1900-01-01")) >= pd.to_datetime(adm.get("admission_date", "1900-01-01")) 
                      and pd.to_datetime(rx.get("start_date", "1900-01-01")) <= pd.to_datetime(adm.get("discharge_date", "2100-01-01")))
            ])

            # Calculate total and average length of stay
            if admissions:
                total_los = sum(adm.get("length_of_stay", 0) for adm in admissions)
                features["total_los_days"] = total_los
                features["avg_los_days"] = total_los / len(admissions)
            else:
                features["total_los_days"] = 0
                features["avg_los_days"] = 0

            # Process prescriptions
            prescriptions = med_hist.get("prescriptions", [])

            # Calculate opioid-related counts and flags
            features["opioid_rx_count"] = sum(
                1 for rx in prescriptions if rx.get("is_opioid", False)
            )

            # Get unique opioids
            features["distinct_opioids"] = len(
                set(
                    rx["drug_name"]
                    for rx in prescriptions
                    if rx.get("is_opioid", False)
                )
            )

            # Calculate opioid exposure days
            opioid_days = sum(
                (pd.to_datetime(rx["end_date"]) - pd.to_datetime(rx["start_date"])).days
                for rx in prescriptions
                if rx.get("is_opioid", False) and "end_date" in rx
            )
            features["opioid_exposure_days"] = max(0, opioid_days)

            # Set flags (binary values)
            features["any_opioid_flag"] = 1 if features["opioid_rx_count"] > 0 else 0
            
            # Check for benzodiazepines (ATC codes N05BA, N05CD, N05CF)
            benzo_codes = ["N05BA", "N05CD", "N05CF"]
            features["any_benzo_flag"] = (
                1 if any(rx.get("drug_class", "").startswith(code) for rx in prescriptions for code in benzo_codes)
                else 0
            )

            # Initialize all ATC class counts to 0
            features["atc_A_rx_count"] = 0
            features["atc_B_rx_count"] = 0
            features["atc_C_rx_count"] = 0
            features["atc_Other_rx_count"] = 0  # All other ATC classes combined

            # Count prescriptions by ATC class
            for rx in prescriptions:
                atc_class = rx.get("drug_class", "")
                if atc_class:
                    atc_prefix = atc_class[0]  # Get first letter of ATC code
                    if atc_prefix == "A":
                        features["atc_A_rx_count"] += 1
                    elif atc_prefix == "B":
                        features["atc_B_rx_count"] += 1
                    elif atc_prefix == "C":
                        features["atc_C_rx_count"] += 1
                    else:
                        features["atc_Other_rx_count"] += 1

            # Set 'y_oud' to 0 (we can't determine this from patient data)
            features["y_oud"] = 0

            return features

        except Exception as e:
            logger.error(f"Error converting patient data to features: {str(e)}")
            raise ValueError(f"Failed to process patient data: {str(e)}")

    def validate_patient_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate patient data structure

        Args:
            data: Dictionary containing patient data

        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Patient data must be a dictionary")

        missing_fields = self.required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        return True
