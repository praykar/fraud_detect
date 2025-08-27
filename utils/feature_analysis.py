import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureAnalyzer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = KNNImputer()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def analyze_features(self, df, target_col):
        """Complete feature analysis pipeline with progress tracking"""
        
        st.write("🔍 Starting feature analysis...")
        
        # 1. Basic Analysis
        st.write("📊 Analyzing basic statistics...")
        stats = self._analyze_basic_stats(df)
        st.write("Basic Statistics:", stats)
        
        # 2. Missing Values
        st.write("🔍 Checking missing values...")
        missing = self._handle_missing_values(df)
        st.write(f"Found {missing['total_missing']} missing values")
        
        # 3. Categorical Encoding
        st.write("🔄 Encoding categorical variables...")
        df_encoded = self._encode_categorical(df)
        st.write("Encoded categorical variables")
        
        # 4. Feature Correlations
        st.write("📈 Calculating feature correlations...")
        corr_matrix = self._analyze_correlations(df_encoded)
        self._plot_correlation_matrix(corr_matrix)
        
        # 5. Feature Importance
        st.write("⭐ Calculating feature importance...")
        importance = self._calculate_feature_importance(df_encoded, target_col)
        self._plot_feature_importance(importance)
        
        return df_encoded, importance
    
    def _analyze_basic_stats(self, df):
        """Analyze basic dataset statistics"""
        stats = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'samples': len(df)
        }
        return stats
    
    def _handle_missing_values(self, df):
        """Handle missing values with tracking"""
        missing = df.isnull().sum()
        total_missing = missing.sum()
        
        if total_missing > 0:
            st.write("Missing value distribution:")
            st.bar_chart(missing[missing > 0])
        
        return {'total_missing': total_missing, 'missing_by_column': missing}
    
    def _encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            st.write(f"Encoding column: {col}")
            self.label_encoders[col] = LabelEncoder()
            df_encoded[col] = self.label_encoders[col].fit_transform(df[col].fillna('MISSING'))
        
        return df_encoded
    
    def _analyze_correlations(self, df):
        """Analyze feature correlations"""
        corr_matrix = df.corr()
        return corr_matrix
    
    def _plot_correlation_matrix(self, corr_matrix):
        """Plot correlation matrix"""
        st.write("Feature Correlation Matrix:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    def _calculate_feature_importance(self, df, target_col):
        """Calculate feature importance scores"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        importance = mutual_info_classif(X, y)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _plot_feature_importance(self, importance_df):
        """Plot feature importance scores"""
        st.write("Feature Importance Scores:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        plt.title('Feature Importance (Mutual Information)')
        st.pyplot(fig)

    def analyze_feature_importance(self, X, y):
        """Analyze feature importance using multiple methods"""
        # Fit model
        self.rf_model.fit(X, y)
        
        # Get standard importance
        std_importance = pd.Series(
            self.rf_model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        # Add noise feature
        X_with_noise = X.copy()
        X_with_noise['random_noise'] = np.random.random(len(X))
        noise_model = RandomForestClassifier(n_estimators=100, random_state=42)
        noise_model.fit(X_with_noise, y)
        noise_importance = noise_model.feature_importances_[-1]
        
        # Calculate cumulative importance
        cumulative_importance = std_importance.cumsum()
        
        # Permutation importance
        perm_importance = permutation_importance(self.rf_model, X, y, n_repeats=10)
        perm_importance_mean = pd.Series(
            perm_importance.importances_mean,
            index=X.columns
        ).sort_values(ascending=False)
        
        return {
            'standard_importance': std_importance,
            'cumulative_importance': cumulative_importance,
            'noise_threshold': noise_importance,
            'permutation_importance': perm_importance_mean
        }

    def plot_importance_analysis(self, importance_results):
        """Create plots for feature importance analysis"""
        fig = plt.figure(figsize=(15, 10))
        
        # Standard importance plot
        plt.subplot(2, 2, 1)
        importance_results['standard_importance'].plot(kind='bar')
        plt.axhline(y=importance_results['noise_threshold'], color='r', linestyle='--',
                   label='Noise Threshold')
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        
        # Cumulative importance plot
        plt.subplot(2, 2, 2)
        importance_results['cumulative_importance'].plot()
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        plt.title('Cumulative Importance')
        
        return fig
