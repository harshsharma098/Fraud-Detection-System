"""
Synthetic Transaction Dataset Generator
Creates realistic transaction data with fraud patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import random


class TransactionDataGenerator:
    """Generate synthetic transaction data with fraud patterns"""

    def __init__(self, n_transactions=100000, fraud_rate=0.02, random_state=42):
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.random_state = random_state

        np.random.seed(random_state)
        random.seed(random_state)

        # India center coordinates
        self.india_center_lat = 20.5937
        self.india_center_long = 78.9629

        self.merchant_categories = [
            "grocery", "electronics", "gas", "restaurant",
            "retail", "jewelry", "luxury_goods"
        ]

        self.fraud_types = [
            "card_cloning", "account_takeover",
            "merchant_collusion", "none"
        ]

    def generate_customer_home_locations(self, n_customers=5000):
        home_locations = {}
        for i in range(1, n_customers + 1):
            lat = float(np.random.normal(self.india_center_lat, 5))
            lon = float(np.random.normal(self.india_center_long, 5))
            home_locations[f"CUST_{i:05d}"] = (lat, lon)
        return home_locations

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def generate_transactions(self):
        n_fraud = int(self.n_transactions * self.fraud_rate)
        n_legit = self.n_transactions - n_fraud

        customer_homes = self.generate_customer_home_locations()
        customer_ids = list(customer_homes.keys())

        transactions = []
        start_date = datetime.now() - timedelta(days=30)

        # Hour probability distribution (normalized)
        hour_probs = np.array(
            [0.02] * 6 + [0.05] * 6 + [0.08] * 6 + [0.05] * 6
        )
        hour_probs = hour_probs / hour_probs.sum()

        # ---------------------------
        # Legitimate transactions
        # ---------------------------
        for i in range(n_legit):
            customer_id = random.choice(customer_ids)
            home_lat, home_lon = customer_homes[customer_id]

            amount = min(float(np.random.lognormal(6, 1.5)), 50000)

            merchant_lat = home_lat + float(np.random.normal(0, 2))
            merchant_lon = home_lon + float(np.random.normal(0, 2))

            distance = self.calculate_distance(
                home_lat, home_lon, merchant_lat, merchant_lon
            )

            day_offset = int(random.randint(0, 30))
            hour = int(np.random.choice(range(24), p=hour_probs))
            minute = int(random.randint(0, 59))

            timestamp = start_date + timedelta(
                days=day_offset,
                hours=hour,
                minutes=minute
            )

            transactions.append({
                "transaction_id": f"TXN_{i + 1:08d}",
                "customer_id": customer_id,
                "card_number": f"CARD_{hash(customer_id) % 100000:05d}",
                "timestamp": timestamp.isoformat() + "Z",
                "amount": round(amount, 2),
                "merchant_id": f"MERCHANT_{random.randint(1, 2000):04d}",
                "merchant_category": random.choice(self.merchant_categories),
                "merchant_lat": round(merchant_lat, 4),
                "merchant_long": round(merchant_lon, 4),
                "is_fraud": 0,
                "fraud_type": "none",
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "month": timestamp.month,
                "distance_from_home": round(distance, 2)
            })

        # ---------------------------
        # Fraudulent transactions
        # ---------------------------
        fraud_patterns = {
            "card_cloning": {
                "amount_multiplier": 2.0,
                "distance_multiplier": 5.0,
                "hour_shift": 0,
                "category_bias": ["jewelry", "luxury_goods", "electronics"]
            },
            "account_takeover": {
                "amount_multiplier": 1.5,
                "distance_multiplier": 10.0,
                "hour_shift": -8,
                "category_bias": ["electronics", "luxury_goods"]
            },
            "merchant_collusion": {
                "amount_multiplier": 3.0,
                "distance_multiplier": 1.0,
                "hour_shift": 0,
                "category_bias": ["retail", "grocery"]
            }
        }

        fraud_types = ["card_cloning", "account_takeover", "merchant_collusion"]
        fraud_weights = [0.4, 0.4, 0.2]

        for i in range(n_fraud):
            fraud_type = np.random.choice(fraud_types, p=fraud_weights)
            pattern = fraud_patterns[fraud_type]

            customer_id = random.choice(customer_ids)
            home_lat, home_lon = customer_homes[customer_id]

            base_amount = float(np.random.lognormal(6, 1.5))
            amount = min(base_amount * pattern["amount_multiplier"], 100000)

            merchant_lat = home_lat + float(np.random.normal(0, pattern["distance_multiplier"] * 2))
            merchant_lon = home_lon + float(np.random.normal(0, pattern["distance_multiplier"] * 2))

            distance = self.calculate_distance(
                home_lat, home_lon, merchant_lat, merchant_lon
            )

            normal_hour = int(np.random.choice(range(24)))
            hour = int((normal_hour + pattern["hour_shift"]) % 24)

            timestamp = start_date + timedelta(
                days=int(random.randint(0, 30)),
                hours=hour,
                minutes=int(random.randint(0, 59)),
                seconds=int(random.randint(0, 59))
            )

            if random.random() < 0.7:
                merchant_category = random.choice(pattern["category_bias"])
            else:
                merchant_category = random.choice(self.merchant_categories)

            transactions.append({
                "transaction_id": f"TXN_{n_legit + i + 1:08d}",
                "customer_id": customer_id,
                "card_number": f"CARD_{hash(customer_id) % 100000:05d}",
                "timestamp": timestamp.isoformat() + "Z",
                "amount": round(amount, 2),
                "merchant_id": f"MERCHANT_{random.randint(1, 2000):04d}",
                "merchant_category": merchant_category,
                "merchant_lat": round(merchant_lat, 4),
                "merchant_long": round(merchant_lon, 4),
                "is_fraud": 1,
                "fraud_type": fraud_type,
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "month": timestamp.month,
                "distance_from_home": round(distance, 2)
            })

        random.shuffle(transactions)
        return pd.DataFrame(transactions)


if __name__ == "__main__":
    generator = TransactionDataGenerator(n_transactions=100000, fraud_rate=0.02)
    df = generator.generate_transactions()

    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print("\nFraud type distribution:")
    print(df[df["is_fraud"] == 1]["fraud_type"].value_counts())

    df.to_csv("data/transactions.csv", index=False)
    print("\nDataset saved to data/transactions.csv")
