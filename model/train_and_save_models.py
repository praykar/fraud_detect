import logging
import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR
from utils.feature_engineering import FeatureEngineer


def create_dummy_data():
    """Create synthetic data for initial model training"""
    np.random.seed(42)
    n_samples = 10000

    # Generate only core features
    features = {
        "amount": np.random.lognormal(mean=4.0, sigma=1.0, size=n_samples),
        "hour": np.random.randint(0, 24, size=n_samples),
        "day_of_week": np.random.randint(0, 7, size=n_samples),
        "merchant_risk_score": np.random.uniform(0, 1, size=n_samples),
        "user_historical_tx_count": np.random.poisson(lam=10, size=n_samples),
    }

    # Create feature matrix
    feature_names = list(features.keys())
    X = np.column_stack([features[f] for f in feature_names])

    # Generate synthetic labels (fraud vs non-fraud)
    y = np.where(
        (features["amount"] > np.percentile(features["amount"], 95))
        & (features["merchant_risk_score"] > 0.8)
        & (features["user_historical_tx_count"] < 5),
        1,
        0,
    )

    # Save feature names with scaler
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, feature_names


def _train_and_evaluate_single_model(name, model, X_train, y_train, X_test, y_test):
    """Trains, evaluates, and saves a single model."""
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"{name} metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    # Save model using absolute path
    model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved {name} model to {model_path}")

    return name, {
        "metrics": metrics,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc},
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_and_save_models(X=None, y=None):
    """
    Train and save models using provided data or generate synthetic data if none provided.

    Args:
        X: Optional feature matrix
        y: Optional target vector
    """
    try:
        print("Starting model training...")
        # Ensure the models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)

        if X is None or y is None:
            X_df, y, feature_names = create_dummy_data()
        else:
            if isinstance(X, np.ndarray):
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=feature_names)
            else:  # Assume it's a DataFrame
                X_df = X
                feature_names = X_df.columns.tolist()

        print(f"Input shapes - X: {X_df.shape}, y: {y.shape}")

        # Engineer and scale features
        feature_engineer = FeatureEngineer()
        X_engineered = feature_engineer.fit_transform(X_df, feature_columns=feature_names)
        print(f"Engineered features shape: {X_engineered.shape}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

        # Train models and collect metrics
        models_to_train = {
            "xgboost": XGBClassifier(n_estimators=100),
            "lightgbm": LGBMClassifier(n_estimators=100),
            "logistic": LogisticRegression(),
        }

        # Use joblib to train models in parallel
        # n_jobs=-1 uses all available CPU cores
        results_list = Parallel(n_jobs=-1)(
            delayed(_train_and_evaluate_single_model)(
                name, model, X_train, y_train, X_test, y_test
            ) for name, model in models_to_train.items()
        )

        # Combine results from the list of tuples into a dictionary
        evaluation_results = {name: result for name, result in results_list}

        # Save the fitted FeatureEngineer instance
        engineer_path = os.path.join(MODELS_DIR, "feature_engineer.pkl")
        joblib.dump(feature_engineer, engineer_path)
        print(f"Saved feature engineer to {engineer_path}")

        # Save evaluation results
        evaluation_path = os.path.join(MODELS_DIR, "evaluation_results.pkl")
        joblib.dump(evaluation_results, evaluation_path)
        print(f"Saved evaluation results to {evaluation_path}")

        return evaluation_results  # Ensure this is returned

    except Exception as e:
        print(f"Error in train_and_save_models: {str(e)}")
        print("Full traceback:", traceback.format_exc())
        return None  # Return None on error


if __name__ == "__main__":
    try:
        train_and_save_models()
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise
