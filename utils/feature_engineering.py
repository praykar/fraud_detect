import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import MODELS_DIR


class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = KNNImputer()
        self.feature_columns_ = None
        self.amount_column_ = None
        self.interaction_pairs_ = []
        self.final_columns_ = None
        self.categorical_columns_ = None
        self.numeric_columns_ = None

    def fit_transform(self, df, feature_columns, amount_column=None):
        """Fits the engineer and transforms the data."""
        self.feature_columns_ = feature_columns
        self.amount_column_ = amount_column
        X = df[self.feature_columns_].copy()

        self.categorical_columns_ = X.select_dtypes(include=['object']).columns.tolist()
        self.numeric_columns_ = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Handle categorical variables
        for col in self.categorical_columns_:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].fillna('MISSING'))

        # Handle numeric variables
        if self.numeric_columns_:
            X[self.numeric_columns_] = self.imputer.fit_transform(X[self.numeric_columns_])

        if self.amount_column_ and self.amount_column_ in df.columns:
            X['amount_zscore'] = self._calculate_zscore(df[amount_column])

        current_numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.interaction_pairs_ = []
        if len(current_numeric_cols) > 1:
            for i in range(len(current_numeric_cols)):
                for j in range(i + 1, len(current_numeric_cols)):
                    self.interaction_pairs_.append((current_numeric_cols[i], current_numeric_cols[j]))

        X = self._add_interaction_features(X)
        self.final_columns_ = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)

        return pd.DataFrame(X_scaled, columns=self.final_columns_)

    def transform(self, df):
        """Transforms new data using the fitted engineer."""
        if self.final_columns_ is None:
            raise RuntimeError("FeatureEngineer has not been fitted. Call fit_transform first.")

        X = df[self.feature_columns_].copy()

        for col in self.categorical_columns_:
            if col in self.label_encoders:
                known_labels = set(self.label_encoders[col].classes_)
                X[col] = X[col].apply(lambda x: x if x in known_labels else 'MISSING')
                X[col] = self.label_encoders[col].transform(X[col].fillna('MISSING'))

        if self.numeric_columns_:
            X[self.numeric_columns_] = self.imputer.transform(X[self.numeric_columns_])

        X = self._add_interaction_features(X)
        X = X.reindex(columns=self.final_columns_, fill_value=0)
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.final_columns_)

    def get_feature_importance(self, X, y):
        """Calculate feature importance"""
        importance = mutual_info_classif(X, y)
        return pd.Series(importance, index=X.columns).sort_values(ascending=False)
    
    def detect_anomalies(self, X):
        """Detect anomalies using Isolation Forest"""
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        return iso_forest.fit_predict(X) == -1
    
    def train_and_save_model(self, X, y):
        """Train and save the model with custom data"""
        try:
            # Save preprocessors
            joblib.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': X.columns.tolist()
            }, os.path.join(MODELS_DIR, 'preprocessors.pkl'))
            
            # Convert DataFrame to numpy array if needed
            X_values = X.values if hasattr(X, 'values') else X
            y_values = y.values if hasattr(y, 'values') else y
            
            # Train models and get evaluation results
            from model.train_and_save_models import train_and_save_models
            evaluation_results = train_and_save_models(X_values, y_values)
            
            if evaluation_results is None:
                raise ValueError("Model training failed to return evaluation results")
                
            return evaluation_results
            
        except Exception as e:
            print(f"Error in train_and_save_model: {e}")
            return None

    def _calculate_zscore(self, series):
        """Calculate z-score for amount"""
        return (series - series.mean()) / series.std()

    def _add_interaction_features(self, X):
        """Add interaction features based on fitted pairs."""
        for col1, col2 in self.interaction_pairs_:
            if col1 in X.columns and col2 in X.columns:
                X[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
        return X
