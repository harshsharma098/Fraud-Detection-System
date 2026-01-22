"""
Pytest configuration and fixtures for API tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_transaction():
    """Fixture providing a valid sample transaction"""
    return {
        "transaction_id": "TXN_TEST_001",
        "customer_id": "CUST_00001",
        "card_number": "CARD_12345",
        "timestamp": "2025-01-15T14:32:45Z",
        "amount": 5000.0,
        "merchant_id": "MERCHANT_1234",
        "merchant_category": "grocery",
        "merchant_lat": 28.5355,
        "merchant_long": 77.3910,
        "hour": 14,
        "day_of_week": 2,
        "month": 1,
        "distance_from_home": 12.5
    }


@pytest.fixture
def high_risk_transaction():
    """Fixture providing a high-risk transaction"""
    return {
        "transaction_id": "TXN_HIGH_RISK",
        "customer_id": "CUST_00001",
        "card_number": "CARD_12345",
        "timestamp": "2025-01-15T03:00:00Z",
        "amount": 50000.0,
        "merchant_id": "MERCHANT_1234",
        "merchant_category": "luxury_goods",
        "merchant_lat": 28.5355,
        "merchant_long": 77.3910,
        "hour": 3,
        "day_of_week": 2,
        "month": 1,
        "distance_from_home": 250.0
    }


@pytest.fixture
def low_risk_transaction():
    """Fixture providing a low-risk transaction"""
    return {
        "transaction_id": "TXN_LOW_RISK",
        "customer_id": "CUST_00001",
        "card_number": "CARD_12345",
        "timestamp": "2025-01-15T14:32:45Z",
        "amount": 500.0,
        "merchant_id": "MERCHANT_1234",
        "merchant_category": "grocery",
        "merchant_lat": 28.5355,
        "merchant_long": 77.3910,
        "hour": 14,
        "day_of_week": 2,
        "month": 1,
        "distance_from_home": 5.0
    }

