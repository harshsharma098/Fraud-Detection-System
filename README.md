# Fraud Detection System

Real-time fraud detection API using Isolation Forest, One-Class SVM, and Autoencoder models.

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python src/data_generator.py
```

### 3. Train Models

```bash
python src/train.py
```

### 4. Start the Application

**Option 1: Streamlit UI (Recommended)**
```bash
streamlit run app.py
```

**Option 2: FastAPI only**
```bash
python src/api.py
```

## Project Structure

```
.
├── app.py                  # Streamlit UI application
├── src/
│   ├── api.py              # FastAPI application
│   ├── models.py           # ML models (Isolation Forest, One-Class SVM, Autoencoder)
│   ├── feature_engineering.py
│   ├── data_generator.py
│   └── train.py
├── config/
│   └── config.yaml         # Configuration
├── data/                   # Dataset
├── models/                 # Trained models
└── logs/                   # API logs
```

## Usage

### Streamlit UI (Recommended)
Run `streamlit run app.py` and use the interactive web interface.

### API Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict fraud probability
- `GET /docs` - API documentation (Swagger UI)

## Model Performance

- **One-Class SVM**: Best overall (F1: 0.59, AUC: 0.86)
- **Isolation Forest**: Good for real-time (F1: 0.18, AUC: 0.80)
- **Autoencoder**: High precision (F1: 0.06, AUC: 0.87)

## Fraud Detection Criteria

**High Risk (≥70%)**: High amount (>₹20K), far distance (>100km), unusual hours, luxury categories

**Moderate Risk (30-70%)**: Some risk factors present

**Low Risk (<30%)**: Normal transaction patterns

## Example Request

```bash
curl -X POST "http://localhost:8000/predict?model_type=isolation_forest" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_001",
    "customer_id": "CUST_001",
    "card_number": "CARD_123",
    "timestamp": "2025-01-15T14:32:45Z",
    "amount": 5000.0,
    "merchant_id": "MERCHANT_123",
    "merchant_category": "grocery",
    "merchant_lat": 28.5355,
    "merchant_long": 77.3910,
    "hour": 14,
    "day_of_week": 2,
    "month": 1,
    "distance_from_home": 12.5
  }'
```

## Configuration

Edit `config/config.yaml` to adjust:
- Model parameters
- API settings
- Feature selection
