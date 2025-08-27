import os
import sys
from datetime import datetime, time

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TransactionProcessor:
    def __init__(self):
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        try:
            scaler_data = joblib.load(scaler_path)
            self.scaler = scaler_data['scaler']
            self.feature_names = scaler_data['feature_names']
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Please run train_and_save_models.py first."
            )
    
    def preprocess_transaction(self, transaction):
        """Convert raw transaction to model features"""
        timestamp = self._get_timestamp(transaction['timestamp'])
        
        # Extract only core features
        features_dict = {
            'amount': float(transaction['amount']),
            'hour': int(timestamp.hour),
            'day_of_week': int(timestamp.weekday()),
            'merchant_risk_score': float(self._get_merchant_risk(transaction['merchant'])),
            'user_historical_tx_count': int(self._get_user_history(transaction['user_id']))
        }
        
        # Ensure correct feature ordering
        feature_array = np.array([[features_dict[f] for f in self.feature_names]])
        
        # Validate feature dimensions
        if feature_array.shape[1] != len(self.feature_names):
            raise ValueError(f"Feature shape mismatch, expected: {len(self.feature_names)}, got {feature_array.shape[1]}")
        
        scaled_features = self.scaler.transform(feature_array)
        return scaled_features

    def _get_timestamp(self, timestamp):
        """Convert timestamp to datetime object"""
        if isinstance(timestamp, time):
            # Use current date with provided time
            current_date = datetime.now().date()
            timestamp = datetime.combine(current_date, timestamp)
        return timestamp

    def _get_merchant_risk(self, merchant):
        # Mock merchant risk calculation
        return np.random.uniform(0, 0.3)
    
    def _get_user_history(self, user_id):
        # Mock user history count
        return np.random.randint(1, 100)
    
    def _get_country_risk(self, country):
        risk_scores = {
            'USA': 0.1,
            'UK': 0.1,
            'Canada': 0.1,
            'Other': 0.5
        }
        return risk_scores.get(country, 0.5)
    
    def _get_device_risk(self, device):
        risk_scores = {
            'Mobile': 0.2,
            'Desktop': 0.1,
            'Tablet': 0.3
        }
        return risk_scores.get(device, 0.5)

def process_input_features(input_values, feature_config):
    """Process input values according to saved feature configuration"""
    try:
        print(f"Processing input type: {type(input_values)}")
        print(f"Input values: {input_values}")
        
        # Convert to DataFrame
        if isinstance(input_values, pd.DataFrame):
            df = input_values.copy()
        elif isinstance(input_values, (dict, list)):
            df = pd.DataFrame([input_values])
        else:
            df = pd.DataFrame(input_values)
            
        print(f"Converted DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Ensure numeric types
        for col in df.columns:
            if col in feature_config.get('numeric_features', []):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Feature processing error: {str(e)}")
        raise

