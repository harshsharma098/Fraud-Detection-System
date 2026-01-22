"""
FastAPI Application for Fraud Detection
Provides /predict endpoint for real-time fraud detection
"""

import yaml
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import joblib
import tensorflow as tf

from src.feature_engineering import FeatureEngineer
from src.models import IsolationForestModel, OneClassSVMModel, AutoencoderModel


# Configure logging
def setup_logging(log_file='logs/api.log'):
    """Setup logging configuration"""
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Load configuration
def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time anomaly detection for payment transactions",
    version="1.0.0"
)

# Static files removed - using Streamlit UI instead

# Global variables for models and feature engineer
models = {}  # Store all models: {'isolation_forest': model, 'one_class_svm': model, 'autoencoder': model}
feature_engineer = None
default_model_type = None


# Request/Response models
class TransactionRequest(BaseModel):
    """Transaction data model for API requests"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    card_number: str = Field(..., description="Card number")
    timestamp: str = Field(..., description="Transaction timestamp (ISO 8601)")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category")
    merchant_lat: float = Field(..., ge=-90, le=90, description="Merchant latitude")
    merchant_long: float = Field(..., ge=-180, le=180, description="Merchant longitude")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in km")
    
    @validator('merchant_category')
    def validate_merchant_category(cls, v):
        valid_categories = [
            'grocery', 'electronics', 'gas', 'restaurant', 
            'retail', 'jewelry', 'luxury_goods'
        ]
        if v not in valid_categories:
            raise ValueError(f"merchant_category must be one of {valid_categories}")
        return v


class PredictionResponse(BaseModel):
    """Response model for fraud predictions"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    model_type: str = Field(..., description="Model used for prediction")
    reasoning: str = Field(..., description="Explanation of the prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


def load_models():
    """Load all trained models and feature engineer"""
    global models, feature_engineer, default_model_type
    
    try:
        # Load feature engineer
        feature_engineer = FeatureEngineer(config)
        feature_engineer.load('models/feature_engineer.joblib')
        logger.info("Feature engineer loaded successfully")
        
        # Load all models
        default_model_type = config['api']['model_type']
        
        # Load Isolation Forest
        try:
            if_model = IsolationForestModel(config)
            if_model.load('models/isolation_forest_model.joblib')
            models['isolation_forest'] = if_model
            logger.info("Isolation Forest model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Isolation Forest: {str(e)}")
        
        # Load One-Class SVM
        try:
            svm_model = OneClassSVMModel(config)
            svm_model.load('models/one_class_svm_model.joblib')
            models['one_class_svm'] = svm_model
            logger.info("One-Class SVM model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load One-Class SVM: {str(e)}")
        
        # Load Autoencoder
        try:
            ae_model = AutoencoderModel(config)
            ae_model.load('models/autoencoder_model.h5')
            models['autoencoder'] = ae_model
            logger.info("Autoencoder model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Autoencoder: {str(e)}")
        
        if not models:
            raise ValueError("No models could be loaded. Please train models first.")
        
        logger.info(f"Loaded {len(models)} model(s): {list(models.keys())}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models on application startup"""
    logger.info("Starting Fraud Detection API...")
    load_models()
    logger.info("API ready to accept requests")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict fraud probability for a transaction",
            "/health": "GET - Health check endpoint",
            "/docs": "GET - API documentation (Swagger UI)"
        },
        "ui": "Use Streamlit UI: streamlit run app.py"
    }




@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "default_model": default_model_type,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def generate_reasoning(fraud_probability: float, transaction_data: dict, model_type: str) -> str:
    """Generate human-readable reasoning for the prediction based on fraud criteria"""
    reasons = []
    risk_factors = []
    
    # Risk criteria based on transaction patterns
    amount = transaction_data['amount']
    distance = transaction_data['distance_from_home']
    category = transaction_data['merchant_category']
    hour = transaction_data['hour']
    
    # High-risk indicators
    if amount > 30000:
        risk_factors.append("Very high transaction amount (₹{:.2f})".format(amount))
    elif amount > 20000:
        risk_factors.append("High transaction amount (₹{:.2f})".format(amount))
    elif amount > 10000:
        risk_factors.append("Above-average transaction amount (₹{:.2f})".format(amount))
    
    if distance > 200:
        risk_factors.append("Transaction location is very far from home ({:.1f} km)".format(distance))
    elif distance > 100:
        risk_factors.append("Transaction location is far from home ({:.1f} km)".format(distance))
    elif distance > 50:
        risk_factors.append("Transaction location is moderately far from home ({:.1f} km)".format(distance))
    
    if category in ['jewelry', 'luxury_goods']:
        risk_factors.append("High-value merchant category: {}".format(category))
    elif category in ['electronics'] and amount > 15000:
        risk_factors.append("High-value electronics purchase")
    
    if hour < 5 or hour > 23:
        risk_factors.append("Transaction occurred during unusual hours ({:02d}:00)".format(hour))
    elif (hour < 8 or hour > 21) and amount > 10000:
        risk_factors.append("Transaction during off-peak hours with high amount")
    
    # Generate reasoning based on probability and risk factors
    if fraud_probability >= 0.7:
        reasons.append("HIGH FRAUD RISK detected by the model")
        if risk_factors:
            reasons.append("Risk factors identified: " + "; ".join(risk_factors))
        else:
            reasons.append("Anomalous patterns detected in transaction features")
        reasons.append("Recommendation: Block transaction and require additional verification")
    elif fraud_probability >= 0.5:
        reasons.append("MODERATE FRAUD RISK detected")
        if risk_factors:
            reasons.append("Risk factors: " + "; ".join(risk_factors))
        reasons.append("Recommendation: Flag for manual review")
    elif fraud_probability >= 0.3:
        reasons.append("LOW-MODERATE RISK")
        if risk_factors:
            reasons.append("Some risk factors present: " + "; ".join(risk_factors[:2]))
        reasons.append("Recommendation: Monitor transaction")
    else:
        reasons.append("LOW RISK - Transaction appears legitimate")
        if not risk_factors:
            reasons.append("No significant risk factors identified")
        else:
            reasons.append("Minor risk factors present but within normal range")
    
    reasoning = f"Using {model_type.replace('_', ' ').title()}: " + ". ".join(reasons) + "."
    return reasoning


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: TransactionRequest,
    model_type: Optional[str] = Query(None, description="Model type: isolation_forest, one_class_svm, or autoencoder")
):
    """
    Predict fraud probability for a transaction
    
    Accepts transaction features and returns fraud probability with reasoning.
    Optionally accepts a model_type query parameter to select which model to use.
    """
    try:
        logger.info(f"Received prediction request for transaction: {request.transaction_id}")
        
        # Select model to use
        selected_model_type = model_type or default_model_type
        
        if selected_model_type not in models:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{selected_model_type}' not available. Available models: {available_models}"
            )
        
        selected_model = models[selected_model_type]
        
        # Convert request to DataFrame
        transaction_dict = request.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Add timestamp column if needed
        if 'timestamp' not in df.columns:
            df['timestamp'] = transaction_dict['timestamp']
        
        # Transform features
        X = feature_engineer.transform(df)
        
        # Predict
        fraud_probability = selected_model.predict_proba(X)[0]
        # Use adaptive threshold based on model type
        # Lower threshold for better recall (catch more fraud)
        threshold = 0.3  # More sensitive threshold
        is_fraud = fraud_probability >= threshold
        
        # Generate reasoning
        reasoning = generate_reasoning(fraud_probability, transaction_dict, selected_model_type)
        
        # Log prediction
        logger.info(
            f"Transaction {request.transaction_id}: "
            f"fraud_prob={fraud_probability:.4f}, "
            f"prediction={'FRAUD' if is_fraud else 'LEGITIMATE'}, "
            f"model={selected_model_type}"
        )
        
        response = PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=float(fraud_probability),
            is_fraud=bool(is_fraud),
            model_type=selected_model_type.replace('_', ' ').title(),
            reasoning=reasoning,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=config['api']['host'],
        port=config['api']['port'],
        log_level=config['api']['log_level'].lower()
    )

