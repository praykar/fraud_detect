import logging
import os
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap

from config import MODELS_DIR
from model.train_and_save_models import *


class FraudDetector:
    def __init__(self):
        self.models_dir = MODELS_DIR
        
        # Initialize models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        if not self._check_models_exist():
            logging.info("Models not found. Training initial models...")
            train_and_save_models()
        
        # Store paths, don't load models yet
        self.model_files = {
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'logistic': 'logistic_model.pkl'
        }
        self.models: Dict[str, Any] = {}  # This will store loaded models
        self.feature_engineer: Any = None  # Lazy load this too
        self.explainer: Any = None  # Lazy load the explainer

        self.weights = {
            'xgboost': 0.4,
            'lightgbm': 0.4,
            'logistic': 0.2
        }
    
    def _check_models_exist(self):
        required_files = [
            'xgboost_model.pkl',
            'lightgbm_model.pkl',
            'logistic_model.pkl',
            'feature_engineer.pkl'
        ]
        return all(os.path.exists(os.path.join(self.models_dir, f)) for f in required_files)
    
    def _load_model(self, name: str) -> Any:
        """Loads a single model from disk if not already loaded."""
        if name not in self.models:
            filename = self.model_files[name]
            path = os.path.join(self.models_dir, filename)
            self.models[name] = joblib.load(path)
        return self.models[name]

    def _load_feature_engineer_if_needed(self) -> Any:
        """Loads the feature engineer from disk if not already loaded."""
        if self.feature_engineer is None:
            try:
                engineer_path = os.path.join(self.models_dir, 'feature_engineer.pkl')
                if os.path.exists(engineer_path):
                    self.feature_engineer = joblib.load(engineer_path)
            except Exception as e:
                print(f"Error loading feature engineer: {e}")
        return self.feature_engineer
    
    def predict(self, features):
        """
        Predict fraud probability for given features
        Args:
            features: numpy array, DataFrame, dict, list, or tuple of features
        """
        try:
            if isinstance(features, dict):
                features_df = pd.DataFrame([features])
            elif isinstance(features, pd.DataFrame):
                features_df = features
            else:
                raise ValueError(f"Unsupported feature type: {type(features)}")

            # Apply feature engineering and scaling
            if self._load_feature_engineer_if_needed():
                processed_features = self.feature_engineer.transform(features_df)
            else:
                raise RuntimeError("Feature engineer not loaded.")
                
            # Get predictions
            predictions = []
            for name in self.model_files.keys():
                try:
                    pred = self._load_model(name).predict_proba(processed_features)[0][1]
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error in model {name}: {e}")
                    
            return np.mean(predictions) if predictions else 0.0
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
    
    def get_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get SHAP values for feature importance"""
        if self._load_feature_engineer_if_needed():
            # The model was trained on processed features, so we need to process
            # the input features before calculating SHAP values.
            processed_features = self.feature_engineer.transform(features)
        else:
            raise RuntimeError("Feature engineer not loaded.")

        if self.explainer is None:
            xgboost_model = self._load_model('xgboost')
            self.explainer = shap.TreeExplainer(xgboost_model)

        shap_values = self.explainer.shap_values(processed_features)

        # For a single prediction, shap_values is a 2D array of shape (1, n_features)
        shap_values_instance = shap_values[0]

        # Try to get feature names from the feature engineer pipeline
        try:
            feature_names = self.feature_engineer.get_feature_names_out()
        except AttributeError:
            # Fallback if the feature engineer doesn't have get_feature_names_out
            feature_names = [f"feature_{i}" for i in range(len(shap_values_instance))]

        importance = dict(zip(feature_names, np.abs(shap_values_instance)))
        
        return importance
