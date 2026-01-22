"""
Streamlit UI for Fraud Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.feature_engineering import FeatureEngineer
from src.models import IsolationForestModel, OneClassSVMModel, AutoencoderModel
import yaml

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_models(config):
    """Load all models and feature engineer"""
    # Load feature engineer
    feature_engineer = FeatureEngineer(config)
    feature_engineer.load('models/feature_engineer.joblib')
    
    # Load models
    models = {}
    
    try:
        if_model = IsolationForestModel(config)
        if_model.load('models/isolation_forest_model.joblib')
        models['isolation_forest'] = if_model
    except FileNotFoundError:
        st.warning("Isolation Forest model file not found. Please train the model first.")
    except Exception as e:
        st.warning(f"Failed to load Isolation Forest: {e}")
    
    try:
        svm_model = OneClassSVMModel(config)
        svm_model.load('models/one_class_svm_model.joblib')
        models['one_class_svm'] = svm_model
    except FileNotFoundError:
        st.warning("One-Class SVM model file not found. Please train the model first.")
    except Exception as e:
        st.warning(f"Failed to load One-Class SVM: {e}")
    
    try:
        ae_model = AutoencoderModel(config)
        ae_model.load('models/autoencoder_model.h5')
        models['autoencoder'] = ae_model
    except FileNotFoundError:
        st.warning("Autoencoder model file not found. Please train the model first.")
    except Exception as e:
        st.warning(f"Failed to load Autoencoder: {e}")
    
    return models, feature_engineer

def generate_reasoning(fraud_probability, transaction_data):
    """Generate human-readable reasoning"""
    reasons = []
    risk_factors = []
    
    amount = transaction_data['amount']
    distance = transaction_data['distance_from_home']
    category = transaction_data['merchant_category']
    hour = transaction_data['hour']
    
    # Risk factors
    if amount > 30000:
        risk_factors.append(f"Very high transaction amount (₹{amount:,.2f})")
    elif amount > 20000:
        risk_factors.append(f"High transaction amount (₹{amount:,.2f})")
    elif amount > 10000:
        risk_factors.append(f"Above-average transaction amount (₹{amount:,.2f})")
    
    if distance > 200:
        risk_factors.append(f"Transaction location is very far from home ({distance:.1f} km)")
    elif distance > 100:
        risk_factors.append(f"Transaction location is far from home ({distance:.1f} km)")
    elif distance > 50:
        risk_factors.append(f"Transaction location is moderately far from home ({distance:.1f} km)")
    
    if category in ['jewelry', 'luxury_goods']:
        risk_factors.append(f"High-value merchant category: {category}")
    elif category in ['electronics'] and amount > 15000:
        risk_factors.append("High-value electronics purchase")
    
    if hour < 5 or hour > 23:
        risk_factors.append(f"Transaction occurred during unusual hours ({hour:02d}:00)")
    elif (hour < 8 or hour > 21) and amount > 10000:
        risk_factors.append("Transaction during off-peak hours with high amount")
    
    # Generate reasoning
    if fraud_probability >= 0.7:
        reasons.append("**HIGH FRAUD RISK** detected")
        if risk_factors:
            reasons.append("Risk factors: " + "; ".join(risk_factors))
        reasons.append("**Recommendation:** Block transaction and require additional verification")
    elif fraud_probability >= 0.5:
        reasons.append("**MODERATE FRAUD RISK** detected")
        if risk_factors:
            reasons.append("Risk factors: " + "; ".join(risk_factors))
        reasons.append("**Recommendation:** Flag for manual review")
    elif fraud_probability >= 0.3:
        reasons.append("**LOW-MODERATE RISK**")
        if risk_factors:
            reasons.append("Some risk factors present: " + "; ".join(risk_factors[:2]))
        reasons.append("**Recommendation:** Monitor transaction")
    else:
        reasons.append("**LOW RISK** - Transaction appears legitimate")
        if not risk_factors:
            reasons.append("No significant risk factors identified")
        else:
            reasons.append("Minor risk factors present but within normal range")
    
    return ". ".join(reasons) + "."

# Main app
def main():
    st.title("🛡️ Fraud Detection System")
    st.markdown("Real-time Transaction Anomaly Detection")
    
    # Load config and models
    config = load_config()
    models, feature_engineer = load_models(config)
    
    if not models:
        st.error("No models loaded. Please train models first.")
        st.stop()
    
    st.sidebar.header("Configuration")
    model_type = st.sidebar.selectbox(
        "Select Model",
        options=list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        index=0
    )
    
    selected_model = models[model_type]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.info(f"**{selected_model.name}** selected")
    
    # Main form
    st.header("Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_id = st.text_input("Transaction ID", value="TXN_00000001")
        customer_id = st.text_input("Customer ID", value="CUST_00001")
        card_number = st.text_input("Card Number", value="CARD_12345")
        amount = st.number_input("Amount (INR)", min_value=0.0, value=5000.0, step=100.0)
        merchant_id = st.text_input("Merchant ID", value="MERCHANT_1234")
        merchant_category = st.selectbox(
            "Merchant Category",
            options=['grocery', 'electronics', 'gas', 'restaurant', 'retail', 'jewelry', 'luxury_goods']
        )
    
    with col2:
        merchant_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=28.5355, step=0.0001)
        merchant_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=77.3910, step=0.0001)
        
        # Date input
        transaction_date = st.date_input("Transaction Date", value=datetime.now().date())
        
        # Time input in 12-hour AM/PM format
        col_time1, col_time2, col_time3 = st.columns(3)
        with col_time1:
            hour_12 = st.selectbox("Hour", options=list(range(1, 13)), index=11)  # 1-12, default 12
        with col_time2:
            minute = st.selectbox("Minute", options=list(range(0, 60, 5)), index=0)  # 0-59, in 5-min intervals
        with col_time3:
            am_pm = st.selectbox("AM/PM", options=["AM", "PM"], index=1)  # AM or PM
        
        # Convert 12-hour to 24-hour format
        if am_pm == "AM":
            hour = 0 if hour_12 == 12 else hour_12
        else:  # PM
            hour = 12 if hour_12 == 12 else hour_12 + 12
        
        # Create time object
        transaction_time = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
        
        distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, value=12.5, step=0.1)
        
        # Extract day_of_week and month from date
        day_of_week = transaction_date.weekday()  # 0=Monday, 6=Sunday
        month = transaction_date.month
    
    # Predict button
    if st.button("🔍 Check for Fraud", type="primary", use_container_width=True):
        # Combine date and time into datetime
        transaction_datetime = datetime.combine(transaction_date, transaction_time)
        
        # Prepare transaction data
        transaction_data = {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'card_number': card_number,
            'timestamp': transaction_datetime.isoformat() + 'Z',
            'amount': amount,
            'merchant_id': merchant_id,
            'merchant_category': merchant_category,
            'merchant_lat': merchant_lat,
            'merchant_long': merchant_long,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'distance_from_home': distance_from_home
        }
        
        try:
            # Show loading indicator
            with st.spinner("Processing transaction..."):
                # Transform features
                df = pd.DataFrame([transaction_data])
                X = feature_engineer.transform(df)
                
                # Predict
                fraud_probability = selected_model.predict_proba(X)[0]
                # Clamp probability to [0.0, 1.0] for display
                fraud_probability = max(0.0, min(1.0, fraud_probability))
                is_fraud = fraud_probability >= 0.3
            
            # Display results
            st.header("Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Probability", f"{fraud_probability:.2%}")
            
            with col2:
                risk_level = "HIGH" if fraud_probability >= 0.7 else "MODERATE" if fraud_probability >= 0.3 else "LOW"
                st.metric("Risk Level", risk_level)
            
            with col3:
                status = "⚠️ FRAUD" if is_fraud else "✓ SAFE"
                st.metric("Status", status)
            
            # Progress bar (clamped to [0.0, 1.0])
            st.progress(fraud_probability)
            
            # Risk indicator
            if fraud_probability >= 0.7:
                st.error("🚨 High Fraud Risk Detected!")
            elif fraud_probability >= 0.3:
                st.warning("⚠️ Moderate Fraud Risk")
            else:
                st.success("✅ Low Risk - Transaction appears legitimate")
            
            # Reasoning
            st.subheader("Analysis & Reasoning")
            reasoning = generate_reasoning(fraud_probability, transaction_data)
            st.info(reasoning)
            
            # Transaction details
            with st.expander("Transaction Details"):
                st.json(transaction_data)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(e)
    
    # Model comparison section
    st.markdown("---")
    st.header("Model Comparison")
    
    if st.button("Compare All Models", use_container_width=True):
        # Combine date and time into datetime
        transaction_datetime = datetime.combine(transaction_date, transaction_time)
        
        # Prepare transaction data
        transaction_data = {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'card_number': card_number,
            'timestamp': transaction_datetime.isoformat() + 'Z',
            'amount': amount,
            'merchant_id': merchant_id,
            'merchant_category': merchant_category,
            'merchant_lat': merchant_lat,
            'merchant_long': merchant_long,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'distance_from_home': distance_from_home
        }
        
        with st.spinner("Comparing all models..."):
            df = pd.DataFrame([transaction_data])
            X = feature_engineer.transform(df)
            
            results = []
            for model_name, model in models.items():
                try:
                    proba = model.predict_proba(X)[0]
                    # Clamp probability to [0.0, 1.0]
                    proba = max(0.0, min(1.0, proba))
                    results.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Fraud Probability': f"{proba:.2%}",
                        'Risk Level': "HIGH" if proba >= 0.7 else "MODERATE" if proba >= 0.3 else "LOW",
                        'Prediction': "FRAUD" if proba >= 0.3 else "SAFE"
                    })
                except Exception as e:
                    results.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Fraud Probability': "Error",
                        'Risk Level': "N/A",
                        'Prediction': f"Error: {str(e)[:50]}..."
                    })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    # Info section
    st.markdown("---")
    st.header("About the Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Isolation Forest**
        
        Fast, efficient anomaly detection using random forests. 
        Best for real-time predictions with low latency.
        """)
    
    with col2:
        st.markdown("""
        **One-Class SVM**
        
        Kernel-based method that learns the boundary of normal transactions. 
        Good for complex patterns.
        """)
    
    with col3:
        st.markdown("""
        **Autoencoder**
        
        Deep learning approach that learns to reconstruct normal transactions. 
        Detects anomalies through reconstruction error.
        """)

if __name__ == "__main__":
    main()

