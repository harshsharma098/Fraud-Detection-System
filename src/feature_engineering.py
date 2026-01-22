"""
Feature Engineering Module
Handles data preprocessing, feature creation, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class FeatureEngineer:
    """Feature engineering and preprocessing pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_features(self, df):
        """Create additional features from raw data"""
        df = df.copy()
        
        # Convert timestamp to datetime if string
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_per_distance'] = df['amount'] / (df['distance_from_home'] + 1)
        
        # Location features
        df['location_cluster'] = (
            (df['merchant_lat'] // 1) * 100 + (df['merchant_long'] // 1)
        ).astype(int)
        
        # Transaction frequency features (if customer_id available)
        if 'customer_id' in df.columns:
            customer_txns = df.groupby('customer_id').size()
            df['customer_txn_count'] = df['customer_id'].map(customer_txns)
            df['customer_txn_count'] = df['customer_txn_count'].fillna(1)
        
        # Merchant category encoding (will be done in fit_transform)
        return df
    
    def fit_transform(self, df, fit=True):
        """Fit transformers and transform data"""
        df = self.create_features(df)
        
        # Encode categorical features
        categorical_features = ['merchant_category']
        
        if fit:
            for feature in categorical_features:
                if feature in df.columns:
                    le = LabelEncoder()
                    df[feature + '_encoded'] = le.fit_transform(df[feature].astype(str))
                    self.label_encoders[feature] = le
        else:
            for feature in categorical_features:
                if feature in df.columns:
                    if feature in self.label_encoders:
                        le = self.label_encoders[feature]
                        # Handle unseen categories
                        known_classes = set(le.classes_)
                        df[feature + '_encoded'] = df[feature].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in known_classes else -1
                        )
        
        # Select features for modeling
        numerical_features = self.config['features']['numerical'] + [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month_sin', 'month_cos', 'amount_log', 
            'amount_per_distance', 'merchant_category_encoded'
        ]
        
        # Filter to only existing columns
        available_features = [f for f in numerical_features if f in df.columns]
        
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = available_features
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=available_features, index=df.index)
    
    def transform(self, df):
        """Transform data using fitted transformers"""
        return self.fit_transform(df, fit=False)
    
    def save(self, path):
        """Save fitted transformers"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, path)
    
    def load(self, path):
        """Load fitted transformers"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        return self


def prepare_data(config):
    """Prepare train/val/test splits"""
    df = pd.read_csv(config['data']['dataset_path'])
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Prepare features
    X = feature_engineer.fit_transform(df, fit=True)
    y = df[config['features']['target']]
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_split'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    val_size = config['data']['val_split'] / (1 - config['data']['test_split'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=config['data']['random_state'],
        stratify=y_temp
    )
    
    # Save feature engineer
    feature_engineer.save('models/feature_engineer.joblib')
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_engineer': feature_engineer
    }

