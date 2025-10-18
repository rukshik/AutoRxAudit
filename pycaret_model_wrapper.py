"""
Model Wrapper for AutoRxAudit
Handles loading and prediction using PyCaret models
"""

import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import pickle
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyCaretModelWrapper:
    """Wrapper for ML model"""

    def __init__(self, model_path: str = None):
        """
        Initialize PyCaret model wrapper

        Args:
            model_path: Path to saved PyCaret model (required)
        """
        if not model_path:
            raise ValueError("Model path is required for PyCaret model")
        
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the PyCaret model from disk"""
        try:
            logger.info(f"Loading PyCaret model from {self.model_path}")
            from pycaret.classification import load_model
            # Remove .pkl if present since load_model adds it
            clean_path = self.model_path[:-4] if self.model_path.lower().endswith('.pkl') else self.model_path
            self.model = load_model(clean_path)
            logger.info("PyCaret model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PyCaret model: {str(e)}")
            raise ValueError(f"Failed to load model from {self.model_path}: {str(e)}")

    def predict_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate risk prediction using PyCaret model

        Args:
            features: Dictionary containing patient features

        Returns:
            Dictionary with risk prediction results
        """
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            if self.model is None:
                raise ValueError("No model loaded")

            # Make prediction using PyCaret model
            predictions = predict_model(self.model, data=df)
            prediction_score = predictions['prediction_score'].iloc[0]
            risk_score = int(prediction_score * 100)  # Convert probability to 0-100 score
            prediction_label = predictions['prediction_label'].iloc[0]

            # Get feature importance from SHAP values if available
            importance_factors = {}
            if hasattr(self.model, 'feature_importances_'):
                for name, importance in zip(df.columns, self.model.feature_importances_):
                    importance_factors[name] = float(importance)

            return {
                "risk_score": risk_score,
                "confidence": float(prediction_score),
                "model_info": f"PyCaret Model ({self.model_path})",
                "feature_importance": importance_factors,
                "prediction": bool(prediction_label)
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"error": "No model loaded"}

        return {
            "model_type": "RandomForestClassifier",
            "model_path": self.model_path or "demo_model",
            "is_demo": self.model_path is None,
        }
