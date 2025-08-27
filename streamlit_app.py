import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR
from model import inference, train_and_save_models
from utils.feature_analysis import FeatureAnalyzer


@st.cache_resource
def load_detector():
    """Load the FraudDetector model, caching it for the session.
    This prevents reloading the model from disk on every prediction.
    """
    return inference.FraudDetector()

def display_evaluation_results(evaluation_results):
    if evaluation_results is None:
        st.error("No evaluation results available. Model training may have failed.")
        return

    st.subheader("Model Evaluation Results")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Metrics", "ROC Curves", "Confusion Matrices"])

    with tab1:
        # Display metrics
        all_metrics = []
        for model_name, results in evaluation_results.items():
            metrics = results.get("metrics", {})
            all_metrics.append({
                "Model": model_name,
                "Precision": metrics.get("1", {}).get("precision"),
                "Recall": metrics.get("1", {}).get("recall"),
                "F1-Score": metrics.get("1", {}).get("f1-score"),
                "Accuracy": metrics.get("accuracy"),
            })
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics).set_index("Model")
            st.table(metrics_df.style.format("{:.4f}", na_rep="N/A"))

    with tab2:
        # Plot ROC curves
        fig, ax = plt.subplots()
        for model_name, results in evaluation_results.items():
            roc = results.get("roc_curve")
            if roc:
                ax.plot(roc.get("fpr"), roc.get("tpr"), label=f'{model_name} (AUC = {roc.get("auc", 0):.3f})')
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        st.pyplot(fig)

    with tab3:
        # Plot confusion matrices
        cols = st.columns(len(evaluation_results))
        for idx, (model_name, results) in enumerate(evaluation_results.items()):
            with cols[idx]:
                st.write(f"### {model_name}")
                cm = results.get("confusion_matrix")
                if cm is not None:
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                    ax.set_title(f"{model_name} Confusion Matrix")
                    st.pyplot(fig)

def load_feature_config():
    """Load feature configuration if exists"""
    try:
        config_path = os.path.join(MODELS_DIR, 'feature_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading feature configuration: {e}")
        return None

def save_feature_config(feature_config):
    """Save feature configuration"""
    try:
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        config_path = os.path.join(MODELS_DIR, 'feature_config.json')
        with open(config_path, 'w') as f:
            json.dump(feature_config, f)
        st.success("Feature configuration saved successfully")
    except Exception as e:
        st.error(f"Error saving feature configuration: {e}")
        raise

def predict_page():
    st.header("Fraud Detection Prediction")
    
    # Load latest feature configuration
    feature_config = load_feature_config()
    
    if not feature_config:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    # Display last training timestamp
    st.info(f"Using model trained at: {feature_config.get('last_trained', 'Unknown')}")
    
    # Create input form based on trained features
    input_values = {}
    for idx, feature in enumerate(feature_config['features']):
        if feature in feature_config.get('numeric_features', []):
            input_values[feature] = st.number_input(
                f"Enter {feature}", 
                value=0.0,
                help=feature_config.get('feature_descriptions', {}).get(feature, ''),
                key=f"num_input_{idx}"  # Add unique key
            )
        else:
            input_values[feature] = st.text_input(
                f"Enter {feature}",
                help=feature_config.get('feature_descriptions', {}).get(feature, ''),
                key=f"text_input_{idx}"  # Add unique key
            )
    
    if st.button("Predict"):
        with st.spinner("Processing..."):
            detector = load_detector()
            # Pass the raw input dictionary directly to the detector
            prediction = detector.predict(input_values)
            
            # Display results
            st.header("Analysis Results")
            
            # Risk gauge
            st.metric("Fraud Risk Score", f"{prediction:.2%}")
            
            # Decision
            if prediction > 0.8:
                st.error("🚨 High Risk - Transaction Blocked")
            elif prediction > 0.5:
                st.warning("⚠️ Medium Risk - Additional Verification Required")
            else:
                st.success("✅ Low Risk - Transaction Approved")
            
            # Feature importance
            st.subheader("Risk Factors")
            # Note: get_feature_importance might need adjustment if it expects raw features
            processed_df = pd.DataFrame([input_values])
            importance = detector.get_feature_importance(processed_df)
            st.bar_chart(importance)    

def validate_and_train():
    st.header("Model Training & Validation")
    
    uploaded_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
    if uploaded_file is not None:
        st.write("📂 Loading data...")
        df = pd.read_csv(uploaded_file)
        
        # Column mapping
        columns = df.columns.tolist()
        id_col = st.selectbox("Transaction ID Column", columns)
        target_col = st.selectbox("Fraud Label Column (0/1)", columns)
        
        if st.button("Analyze and Train"):
            with st.spinner("Analyzing features and training model..."):
                # Initialize feature analyzer
                analyzer = FeatureAnalyzer()
                
                # Perform feature analysis
                df_processed, feature_importance = analyzer.analyze_features(
                    df.drop(columns=[id_col]), 
                    target_col
                )
                
                # Select top features
                n_features = st.slider("Select number of top features to use", 
                                     min_value=2,
                                     max_value=len(feature_importance),
                                     value=min(10, len(feature_importance)))
                
                top_features = feature_importance['feature'][:n_features].tolist()
                st.write("🔝 Selected top features:", top_features)
                
                # Train model with selected features
                st.write("🚀 Training model...")
                results = train_and_save_models.train_and_save_models(
                    df_processed[top_features],
                    df_processed[target_col]
                )
                
                if results:
                    # Update feature configuration after training
                    feature_config = {
                        'features': top_features,
                        'numeric_features': df[top_features].select_dtypes(include=['int64', 'float64']).columns.tolist(),
                        'last_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'feature_descriptions': {
                            feature: f"Feature importance: {score:.4f}" 
                            for feature, score in zip(feature_importance['feature'], feature_importance['importance'])
                        }
                    }
                    save_feature_config(feature_config)
                    
                    st.success("✅ Model and features updated successfully!")
                    return results
                    # st.experimental_rerun()  # Refresh the page to update prediction interface


def create_timeline():
    """Create horizontal timeline showing progress"""
    timeline_html = """
        <style>
            .timeline {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 20px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .step {
                display: flex;
                flex-direction: column;
                align-items: center;
                flex: 1;
            }
            .step-active { color: #0066cc; font-weight: bold; }
            .step-completed { color: #28a745; }
            .step-pending { color: #6c757d; }
            .connector {
                height: 2px;
                background-color: #dee2e6;
                flex: 1;
                margin: 0 10px;
            }
            .connector-completed { background-color: #28a745; }
        </style>
        """
    return timeline_html

def cleanup_memory():
    """Clean up memory between steps"""
    import gc
    gc.collect()
    plt.close('all')

def transition_to_next_step(current_results=None, wait_time=3):
    """Handle transition to next step"""
    if current_results is not None:
        st.session_state.training_results = current_results
    
    with st.spinner("Processing..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(wait_time/100)
            progress_bar.progress(i + 1)
        st.session_state.current_step += 1
        st.rerun()

def main():
    st.title("Fraud Detection System")
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    
    # Create timeline
    timeline = create_timeline()
    steps = [
        {"title": "Upload & Train", "icon": "📊"},
        {"title": "Model Evaluation", "icon": "📈"},
        {"title": "Make Predictions", "icon": "🎯"}
    ]
    
    # Render timeline
    timeline_html = timeline
    timeline_html += '<div class="timeline">'
    for i, step in enumerate(steps, 1):
        status = ""
        if i < st.session_state.current_step:
            status = "step-completed"
        elif i == st.session_state.current_step:
            status = "step-active"
        else:
            status = "step-pending"
            
        timeline_html += f"""
            <div class="step {status}">
                <div>{step['icon']}</div>
                <div>{step['title']}</div>
            </div>
        """
        if i < len(steps):
            connector_class = "connector-completed" if i < st.session_state.current_step else ""
            timeline_html += f'<div class="connector {connector_class}"></div>'
    timeline_html += '</div>'
    
    st.markdown(timeline_html, unsafe_allow_html=True)
    
    # Display current step content
    if st.session_state.current_step == 1:
        results = validate_and_train()                
        if results:
            # Store evaluation results
            st.session_state.evaluation_results = results
            
            st.success("Training completed! Redirecting to evaluation...")
            cleanup_memory()
            transition_to_next_step()
        
    
    elif st.session_state.current_step == 2:
        st.header("Step 2: Model Evaluation")
        if st.session_state.evaluation_results is not None:
            display_evaluation_results(st.session_state.evaluation_results)
            
            if st.button("Proceed to Predictions"):
                cleanup_memory()
                transition_to_next_step(wait_time=2)
        
        if st.button("← Retrain Model"):
            cleanup_memory()
            st.session_state.current_step = 1
            st.session_state.evaluation_results = None
            st.session_state.feature_importance = None
    
    else:
        st.header("Step 3: Make Predictions")
        predict_page()
        if st.button("← Start Over"):
            cleanup_memory()
            st.session_state.current_step = 1
            st.session_state.evaluation_results = None
            st.session_state.feature_importance = None

def check_model_status():
    """Check if model is trained and get training details"""
    feature_config = load_feature_config()
    return {
        'is_trained': feature_config is not None,
        'last_trained': feature_config.get('last_trained', 'Unknown') if feature_config else None,
        'features': feature_config.get('features', []) if feature_config else []
    }

if __name__ == "__main__":
    main()
