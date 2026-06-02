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


def inject_custom_css():
    """Apply a modern visual theme to the Streamlit app."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(99, 102, 241, 0.18), transparent 26rem),
                radial-gradient(circle at 88% 18%, rgba(14, 165, 233, 0.16), transparent 28rem),
                linear-gradient(135deg, #f8fafc 0%, #eff6ff 48%, #f5f3ff 100%);
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        .main,
        .main p,
        .main span,
        .main label,
        .main h1,
        .main h2,
        .main h3,
        .main h4,
        .main h5,
        .main h6,
        .main div[data-testid="stMarkdownContainer"] {
            color: #111827 !important;
        }

        .main label,
        .main [data-testid="stWidgetLabel"] p {
            color: #334155 !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
        }

        [data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMarkdown p {
            color: #cbd5e1;
        }

        .hero-card {
            padding: 2.25rem;
            border-radius: 28px;
            color: white;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.97), rgba(30, 64, 175, 0.94)),
                radial-gradient(circle at 85% 20%, rgba(125, 211, 252, 0.45), transparent 22rem);
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.28);
            margin-bottom: 1.8rem;
            overflow: hidden;
            position: relative;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.1);
            color: #bfdbfe !important;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .hero-card h1 {
            margin: 1rem 0 0.7rem 0;
            color: #ffffff !important;
            font-size: clamp(2.25rem, 6vw, 4.4rem);
            line-height: 0.95;
            font-weight: 800;
            letter-spacing: -0.06em;
        }

        .hero-card p {
            max-width: 760px;
            margin: 0;
            color: #dbeafe !important;
            font-size: 1.08rem;
            line-height: 1.7;
        }

        .hero-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1.5rem;
        }

        .hero-stat {
            padding: 0.8rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.16);
            backdrop-filter: blur(16px);
            min-width: 150px;
        }

        .hero-stat strong {
            display: block;
            font-size: 1.2rem;
            color: #ffffff !important;
        }

        .hero-stat span {
            color: #bfdbfe !important;
            font-size: 0.82rem;
        }

        .section-title {
            margin: 1.4rem 0 0.45rem;
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .section-subtitle {
            color: #475569 !important;
            margin-bottom: 1.1rem;
            font-weight: 500;
        }

        .glass-panel {
            padding: 1.25rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.28);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .model-card,
        .sidebar-card,
        .result-card {
            padding: 1.1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
        }

        .sidebar-card {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.16);
            box-shadow: none;
        }

        .sidebar-card h3 {
            margin: 0 0 0.35rem;
            color: #ffffff;
        }

        .sidebar-card p {
            margin: 0;
            color: #cbd5e1;
            font-size: 0.92rem;
        }

        .risk-card {
            padding: 1.3rem;
            border-radius: 24px;
            color: white;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
            margin: 1rem 0;
        }

        .risk-high {
            background: linear-gradient(135deg, #991b1b, #ef4444);
        }

        .risk-moderate {
            background: linear-gradient(135deg, #92400e, #f59e0b);
        }

        .risk-low {
            background: linear-gradient(135deg, #065f46, #10b981);
        }

        .risk-card h2,
        .risk-card p {
            margin: 0;
            color: #ffffff !important;
        }

        .risk-card h2 {
            font-size: 2rem;
            letter-spacing: -0.03em;
        }

        .risk-card p {
            margin-top: 0.4rem;
            opacity: 0.92;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        }

        .stButton > button {
            border-radius: 18px;
            min-height: 3rem;
            border: 0;
            color: #ffffff !important;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
            box-shadow: 0 16px 36px rgba(79, 70, 229, 0.3);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .stButton > button:hover {
            border: 0;
            transform: translateY(-1px);
            box-shadow: 0 20px 42px rgba(79, 70, 229, 0.38);
        }

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stDateInput input {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #dbe3ef !important;
            border-radius: 15px !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            min-height: 2.9rem;
        }

        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stDateInput input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.12) !important;
        }

        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] svg {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        .stNumberInput button {
            background: #eef2ff !important;
            color: #4338ca !important;
            border: 1px solid #dbe3ef !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 24px;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.09);
            padding: 1.1rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] h4 {
            color: #0f172a !important;
            margin-bottom: 0.9rem;
            font-size: 1.1rem;
        }

        .model-card h3 {
            color: #0f172a;
            margin-top: 0;
        }

        .model-card p {
            color: #64748b;
            line-height: 1.6;
        }

        /* Strong visibility fixes for the transaction form */
        section.main [data-testid="stWidgetLabel"],
        section.main [data-testid="stWidgetLabel"] *,
        section.main label,
        section.main label *,
        section.main .stTextInput label,
        section.main .stNumberInput label,
        section.main .stSelectbox label,
        section.main .stDateInput label {
            color: #0f172a !important;
            opacity: 1 !important;
            font-weight: 700 !important;
        }

        section.main [data-testid="stMarkdownContainer"] h4,
        section.main [data-testid="stMarkdownContainer"] h3,
        section.main [data-testid="stMarkdownContainer"] p {
            color: #0f172a !important;
            opacity: 1 !important;
        }

        section.main input,
        section.main textarea,
        section.main input::placeholder,
        section.main textarea::placeholder {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            opacity: 1 !important;
        }

        section.main div[data-baseweb="select"],
        section.main div[data-baseweb="select"] *,
        section.main div[data-baseweb="popover"] *,
        section.main [role="option"] {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            opacity: 1 !important;
        }

        section.main input {
            background-color: #ffffff !important;
        }

        section.main .hero-card,
        section.main .hero-card *,
        section.main .risk-card,
        section.main .risk-card * {
            -webkit-text-fill-color: unset !important;
        }

        section.main .hero-card h1,
        section.main .hero-card strong,
        section.main .risk-card h2,
        section.main .risk-card p {
            color: #ffffff !important;
        }

        section.main .hero-card p {
            color: #dbeafe !important;
        }

        section.main .hero-kicker,
        section.main .hero-stat span {
            color: #bfdbfe !important;
        }

        /* AML Watcher-inspired visual refresh */
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(0, 199, 177, 0.16), transparent 28rem),
                radial-gradient(circle at 88% 14%, rgba(56, 189, 248, 0.14), transparent 30rem),
                linear-gradient(135deg, #f6fbff 0%, #eef7f6 46%, #f8fbff 100%) !important;
        }

        .hero-card {
            background:
                radial-gradient(circle at 88% 18%, rgba(45, 212, 191, 0.28), transparent 24rem),
                linear-gradient(135deg, #07111f 0%, #0b1f33 50%, #0f3a45 100%) !important;
            border: 1px solid rgba(45, 212, 191, 0.28);
            box-shadow: 0 28px 80px rgba(7, 17, 31, 0.34) !important;
        }

        .hero-kicker {
            background: rgba(20, 184, 166, 0.14) !important;
            border-color: rgba(94, 234, 212, 0.35) !important;
            color: #99f6e4 !important;
        }

        .hero-stat {
            background: rgba(255, 255, 255, 0.09) !important;
            border-color: rgba(94, 234, 212, 0.24) !important;
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(20, 184, 166, 0.18), transparent 18rem),
                linear-gradient(180deg, #07111f 0%, #0b1f33 100%) !important;
        }

        .sidebar-card {
            background: rgba(255, 255, 255, 0.07) !important;
            border: 1px solid rgba(94, 234, 212, 0.22) !important;
            border-radius: 22px !important;
        }

        .section-title {
            color: #07111f !important;
        }

        .section-subtitle {
            color: #475569 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid #dcebe8 !important;
            border-radius: 26px !important;
            box-shadow: 0 22px 54px rgba(7, 17, 31, 0.1) !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox div[data-baseweb="select"] {
            background: #f8fafc !important;
            border: 1px solid #cbdedb !important;
            border-radius: 14px !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            box-shadow: none !important;
        }

        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stDateInput input:focus,
        .stSelectbox div[data-baseweb="select"]:focus-within {
            background: #ffffff !important;
            border-color: #14b8a6 !important;
            box-shadow: 0 0 0 4px rgba(20, 184, 166, 0.14) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #00b894 0%, #00a8cc 100%) !important;
            color: #ffffff !important;
            border-radius: 999px !important;
            min-height: 3.2rem !important;
            box-shadow: 0 16px 36px rgba(0, 168, 204, 0.28) !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #00a887 0%, #0284c7 100%) !important;
            box-shadow: 0 22px 46px rgba(0, 168, 204, 0.38) !important;
        }

        .stButton {
            margin-top: 1.4rem !important;
            margin-bottom: 1rem !important;
        }

        .stButton > button,
        button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-secondary"],
        div[data-testid="stButton"] button,
        div[data-testid="stButton"] button[kind="primary"] {
            width: 100% !important;
            background: linear-gradient(135deg, #00b894 0%, #00a8cc 100%) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border: 0 !important;
            border-radius: 999px !important;
            min-height: 3.35rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 16px 36px rgba(0, 168, 204, 0.28) !important;
        }

        .stButton > button *,
        button[data-testid="baseButton-primary"] *,
        button[data-testid="baseButton-secondary"] *,
        div[data-testid="stButton"] button * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }

        .stButton > button:hover,
        button[data-testid="baseButton-primary"]:hover,
        button[data-testid="baseButton-secondary"]:hover,
        div[data-testid="stButton"] button:hover {
            background: linear-gradient(135deg, #00a887 0%, #0284c7 100%) !important;
            border: 0 !important;
            transform: translateY(-1px);
            box-shadow: 0 22px 46px rgba(0, 168, 204, 0.38) !important;
        }

        .stButton > button:active,
        .stButton > button:focus,
        button[data-testid="baseButton-primary"]:active,
        button[data-testid="baseButton-primary"]:focus,
        div[data-testid="stButton"] button:active,
        div[data-testid="stButton"] button:focus {
            background: linear-gradient(135deg, #009f82 0%, #0277a8 100%) !important;
            border: 0 !important;
            outline: 4px solid rgba(20, 184, 166, 0.18) !important;
            box-shadow: 0 18px 40px rgba(0, 168, 204, 0.3) !important;
        }

        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #dcebe8 !important;
            box-shadow: 0 16px 34px rgba(7, 17, 31, 0.08) !important;
        }

        .model-card {
            background: #ffffff !important;
            border: 1px solid #dcebe8 !important;
            box-shadow: 0 18px 44px rgba(7, 17, 31, 0.08) !important;
        }

        .model-card h3 {
            color: #07111f !important;
        }

        .risk-high {
            background: linear-gradient(135deg, #7f1d1d, #ef4444) !important;
        }

        .risk-moderate {
            background: linear-gradient(135deg, #92400e, #f59e0b) !important;
        }

        .risk-low {
            background: linear-gradient(135deg, #064e3b, #00b894) !important;
        }

        /* Final override: keep all form headings and field names dark black */
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] h1,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] h2,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] h3,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] h4,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] h5,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] h6,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] p,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] label,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] label *,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"],
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] *,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"],
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] * {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }

        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] h4 {
            font-weight: 800 !important;
        }

        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] label,
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] * {
            font-weight: 700 !important;
        }

        .form-card-title {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            font-size: 1.18rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            margin: 0.15rem 0 1rem;
            opacity: 1 !important;
        }

        .field-label {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            display: block;
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            line-height: 1.2;
            margin: 0.85rem 0 0.35rem;
            opacity: 1 !important;
            text-shadow: none !important;
        }

        [data-testid="stSidebar"] .sidebar-field-label {
            color: #eafffb !important;
            -webkit-text-fill-color: #eafffb !important;
            font-size: 0.9rem;
            font-weight: 800;
            margin: 0.75rem 0 0.4rem;
        }

        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff !important;
            background: #ffffff !important;
            border: 1px solid rgba(153, 246, 228, 0.42) !important;
            color: #07111f !important;
            -webkit-text-fill-color: #07111f !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] *,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] svg {
            color: #07111f !important;
            -webkit-text-fill-color: #07111f !important;
            fill: #07111f !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div[role="button"] {
            background-color: #ffffff !important;
            color: #07111f !important;
            -webkit-text-fill-color: #07111f !important;
        }

        div[data-baseweb="popover"] {
            background: #ffffff !important;
            border: 1px solid #cbdedb !important;
            border-radius: 14px !important;
            box-shadow: 0 18px 44px rgba(7, 17, 31, 0.16) !important;
        }

        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] li *,
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [role="option"] * {
            background: #ffffff !important;
            color: #07111f !important;
            -webkit-text-fill-color: #07111f !important;
            opacity: 1 !important;
        }

        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] [role="option"]:hover {
            background: #e6fffb !important;
        }
    </style>
    """, unsafe_allow_html=True)


def field_label(text):
    """Render a reliable, high-contrast label above Streamlit widgets."""
    st.markdown(f'<div class="field-label">{text}</div>', unsafe_allow_html=True)

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
    inject_custom_css()
    st.markdown("""
    <div class="hero-card">
        <div class="hero-kicker">AI-Powered Fraud Monitoring</div>
        <h1>Real-Time Fraud Intelligence</h1>
        <p>
            Analyze transactions instantly with explainable anomaly detection, intelligent risk scoring,
            and side-by-side model comparison designed for faster fraud review and smarter decisions.
        </p>
        <div class="hero-stats">
            <div class="hero-stat"><strong>3</strong><span>Advanced Detection Models</span></div>
            <div class="hero-stat"><strong>100K</strong><span>Transactions Used for Training</span></div>
            <div class="hero-stat"><strong>Live</strong><span>Real-Time Transaction Screening</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load config and models
    config = load_config()
    models, feature_engineer = load_models(config)
    
    if not models:
        st.error("No models loaded. Please train models first.")
        st.stop()
    
    st.sidebar.markdown("""
    <div class="sidebar-card">
        <h3>Risk Console</h3>
        <p>Select a detection model and review every transaction with confidence.</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("### Configuration")
    st.sidebar.markdown('<div class="sidebar-field-label">Select Model</div>', unsafe_allow_html=True)
    model_type = st.sidebar.selectbox(
        "Select Model",
        options=list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        index=0,
        label_visibility="collapsed"
    )
    
    selected_model = models[model_type]
    
    st.sidebar.markdown("""
    <div class="sidebar-card">
        <h3>Selected Model</h3>
        <p><strong>{}</strong> is active for the main prediction.</p>
    </div>
    """.format(selected_model.name), unsafe_allow_html=True)
    
    # Main form
    st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter transaction context and run a fraud risk check.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown('<div class="form-card-title">Customer & Payment</div>', unsafe_allow_html=True)
            field_label("Transaction ID")
            transaction_id = st.text_input("Transaction ID", value="TXN_00000001", label_visibility="collapsed")
            field_label("Customer ID")
            customer_id = st.text_input("Customer ID", value="CUST_00001", label_visibility="collapsed")
            field_label("Card Number")
            card_number = st.text_input("Card Number", value="CARD_12345", label_visibility="collapsed")
            field_label("Amount (INR)")
            amount = st.number_input("Amount (INR)", min_value=0.0, value=5000.0, step=100.0, label_visibility="collapsed")
            field_label("Merchant ID")
            merchant_id = st.text_input("Merchant ID", value="MERCHANT_1234", label_visibility="collapsed")
            field_label("Merchant Category")
            merchant_category = st.selectbox(
                "Merchant Category",
                options=['grocery', 'electronics', 'gas', 'restaurant', 'retail', 'jewelry', 'luxury_goods'],
                label_visibility="collapsed"
            )
    
    with col2:
        with st.container(border=True):
            st.markdown('<div class="form-card-title">Location & Timing</div>', unsafe_allow_html=True)
            field_label("Merchant Latitude")
            merchant_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=28.5355, step=0.0001, label_visibility="collapsed")
            field_label("Merchant Longitude")
            merchant_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=77.3910, step=0.0001, label_visibility="collapsed")
            
            # Date input
            field_label("Transaction Date")
            transaction_date = st.date_input("Transaction Date", value=datetime.now().date(), label_visibility="collapsed")
            
            # Time input in 12-hour AM/PM format
            col_time1, col_time2, col_time3 = st.columns(3)
            with col_time1:
                field_label("Hour")
                hour_12 = st.selectbox("Hour", options=list(range(1, 13)), index=11, label_visibility="collapsed")  # 1-12, default 12
            with col_time2:
                field_label("Minute")
                minute = st.selectbox("Minute", options=list(range(0, 60, 5)), index=0, label_visibility="collapsed")  # 0-59, in 5-min intervals
            with col_time3:
                field_label("AM/PM")
                am_pm = st.selectbox("AM/PM", options=["AM", "PM"], index=1, label_visibility="collapsed")  # AM or PM
            
            # Convert 12-hour to 24-hour format
            if am_pm == "AM":
                hour = 0 if hour_12 == 12 else hour_12
            else:  # PM
                hour = 12 if hour_12 == 12 else hour_12 + 12
            
            # Create time object
            transaction_time = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
            
            field_label("Distance from Home (km)")
            distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, value=12.5, step=0.1, label_visibility="collapsed")
            
            # Extract day_of_week and month from date
            day_of_week = transaction_date.weekday()  # 0=Monday, 6=Sunday
            month = transaction_date.month
    
    st.markdown('<div style="height: 0.4rem;"></div>', unsafe_allow_html=True)

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
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
            
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
                risk_class = "risk-high"
                risk_title = "High Fraud Risk Detected"
                risk_message = "Block this transaction and request additional verification."
            elif fraud_probability >= 0.3:
                risk_class = "risk-moderate"
                risk_title = "Moderate Fraud Risk"
                risk_message = "Flag this transaction for review before approval."
            else:
                risk_class = "risk-low"
                risk_title = "Low Risk Transaction"
                risk_message = "The transaction appears consistent with normal behavior."
            
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <h2>{risk_title}</h2>
                <p>{risk_message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reasoning
            st.markdown('<div class="section-title">Analysis & Reasoning</div>', unsafe_allow_html=True)
            reasoning = generate_reasoning(fraud_probability, transaction_data)
            st.info(reasoning)
            
            # Transaction details
            with st.expander("Transaction Details"):
                st.json(transaction_data)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(e)
    
    # Model comparison section
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Run the same transaction through every available model.</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-title">About the Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Each model looks at transaction behavior from a different angle.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>Isolation Forest</h3>
            <p>Fast anomaly detection using randomized trees. Best for quick, low-latency transaction screening.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>One-Class SVM</h3>
            <p>Learns the boundary of normal transaction behavior and performs well on complex fraud patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-card">
            <h3>Autoencoder</h3>
            <p>Uses reconstruction error to detect unusual transactions that differ from learned normal behavior.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

