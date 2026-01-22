"""
Anomaly Detection Models
Implements Isolation Forest, One-Class SVM, and Autoencoder
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class IsolationForestModel:
    """Isolation Forest for anomaly detection"""
    
    def __init__(self, config):
        self.config = config
        self.model = IsolationForest(
            contamination=config['model']['isolation_forest']['contamination'],
            n_estimators=config['model']['isolation_forest']['n_estimators'],
            max_samples=config['model']['isolation_forest']['max_samples'],
            random_state=config['model']['isolation_forest']['random_state'],
            n_jobs=-1
        )
        self.name = "Isolation Forest"
        # Calibration parameters for probability conversion
        self.score_min = None
        self.score_max = None
        self.score_percentiles = None
    
    def train(self, X_train, y_train=None):
        """Train the model (unsupervised, y_train ignored)"""
        # For Isolation Forest, we use only normal samples for training
        # But since it's unsupervised, we can use all data
        self.model.fit(X_train)
        
        # Calculate calibration parameters from training data
        train_scores = self.model.score_samples(X_train)
        self.score_min = np.min(train_scores)
        self.score_max = np.max(train_scores)
        # Store percentiles for better probability estimation
        self.score_percentiles = {
            'p1': np.percentile(train_scores, 1),
            'p5': np.percentile(train_scores, 5),
            'p10': np.percentile(train_scores, 10),
            'p25': np.percentile(train_scores, 25),
            'p50': np.percentile(train_scores, 50),
            'p75': np.percentile(train_scores, 75),
            'p90': np.percentile(train_scores, 90),
            'p95': np.percentile(train_scores, 95),
            'p99': np.percentile(train_scores, 99)
        }
        return self
    
    def predict_proba(self, X):
        """Predict fraud probability using calibrated parameters"""
        # Isolation Forest: lower scores = more anomalous = higher fraud probability
        # Typical score range: -0.6 to 0.0 (anomalies have negative scores)
        scores = self.model.score_samples(X)
        predictions = self.model.predict(X)  # -1 for anomaly, 1 for normal
        
        # Use calibration parameters if available
        if self.score_min is not None and self.score_max is not None:
            score_range = self.score_max - self.score_min
            if score_range > 0:
                normalized = (scores - self.score_min) / score_range
                fraud_proba = 1 - normalized  # Lower scores = higher fraud prob
            else:
                # Use prediction directly - be more conservative
                fraud_proba = (predictions == -1).astype(float) * 0.5  # Reduced from 0.7
        else:
            # Fallback: use percentile-based estimation if available
            if self.score_percentiles is not None:
                fraud_proba = np.zeros(len(scores))
                for i, score in enumerate(scores):
                    if score <= self.score_percentiles['p1']:
                        fraud_proba[i] = 0.95
                    elif score <= self.score_percentiles['p5']:
                        fraud_proba[i] = 0.85
                    elif score <= self.score_percentiles['p10']:
                        fraud_proba[i] = 0.70
                    elif score <= self.score_percentiles['p25']:
                        fraud_proba[i] = 0.50
                    elif score <= self.score_percentiles['p50']:
                        fraud_proba[i] = 0.30
                    elif score <= self.score_percentiles['p75']:
                        fraud_proba[i] = 0.15
                    elif score <= self.score_percentiles['p90']:
                        fraud_proba[i] = 0.05
                    else:
                        fraud_proba[i] = 0.02
            else:
                # Smart fallback: Use prediction + score-based estimation
                # Isolation Forest scores are typically negative for anomalies
                # Normalize based on typical ranges - be more conservative
                fraud_proba = np.zeros(len(scores))
                for i, (score, pred) in enumerate(zip(scores, predictions)):
                    if pred == -1:  # Anomaly detected
                        # Score-based probability: more negative = higher fraud
                        # But be more conservative - most anomalies are moderate risk
                        if score < -0.5:
                            fraud_proba[i] = 0.65  # Reduced from 0.90
                        elif score < -0.3:
                            fraud_proba[i] = 0.50  # Reduced from 0.75
                        elif score < -0.1:
                            fraud_proba[i] = 0.40  # Reduced from 0.60
                        else:
                            fraud_proba[i] = 0.30  # Reduced from 0.45
                    else:  # Normal - should be low risk
                        # Normal predictions should have low fraud probability
                        if score < -0.2:
                            fraud_proba[i] = 0.20  # Reduced from 0.35
                        elif score < 0.0:
                            fraud_proba[i] = 0.10  # Reduced from 0.20
                        else:
                            fraud_proba[i] = 0.05  # Keep low
        
        return fraud_proba
    
    def predict(self, X, threshold=0.5):
        """Predict binary fraud labels"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc,
            'confusion_matrix': cm.tolist()
        }
    
    def save(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load(self, path):
        """Load model"""
        self.model = joblib.load(path)
        return self


class OneClassSVMModel:
    """One-Class SVM for anomaly detection"""
    
    def __init__(self, config):
        self.config = config
        self.model = OneClassSVM(
            nu=config['model']['one_class_svm']['nu'],
            kernel=config['model']['one_class_svm']['kernel'],
            gamma=config['model']['one_class_svm']['gamma']
        )
        self.name = "One-Class SVM"
        self.scaler = None
        # Calibration parameters for probability conversion
        self.score_min = None
        self.score_max = None
        self.score_percentiles = None
    
    def train(self, X_train, y_train=None):
        """Train the model on normal samples only"""
        # Use only normal samples for training
        if y_train is not None:
            normal_mask = y_train == 0
            X_normal = X_train[normal_mask]
        else:
            X_normal = X_train
        
        self.model.fit(X_normal)
        
        # Calculate calibration parameters from training data
        train_scores = self.model.decision_function(X_normal)
        self.score_min = np.min(train_scores)
        self.score_max = np.max(train_scores)
        # Store percentiles for better probability estimation
        self.score_percentiles = {
            'p1': np.percentile(train_scores, 1),
            'p5': np.percentile(train_scores, 5),
            'p10': np.percentile(train_scores, 10),
            'p25': np.percentile(train_scores, 25),
            'p50': np.percentile(train_scores, 50),
            'p75': np.percentile(train_scores, 75),
            'p90': np.percentile(train_scores, 90),
            'p95': np.percentile(train_scores, 95),
            'p99': np.percentile(train_scores, 99)
        }
        return self
    
    def predict_proba(self, X):
        """Predict fraud probability using calibrated parameters"""
        # One-Class SVM: lower decision scores = more anomalous = higher fraud probability
        decision_scores = self.model.decision_function(X)
        predictions = self.model.predict(X)  # -1 for anomaly, 1 for normal
        
        # Use calibration parameters if available
        if self.score_min is not None and self.score_max is not None:
            score_range = self.score_max - self.score_min
            if score_range > 0:
                normalized = (decision_scores - self.score_min) / score_range
                fraud_proba = 1 - normalized  # Lower scores = higher fraud prob
            else:
                fraud_proba = (predictions == -1).astype(float) * 0.7
        else:
            # Fallback: use percentile-based estimation if available
            if self.score_percentiles is not None:
                fraud_proba = np.zeros(len(decision_scores))
                for i, score in enumerate(decision_scores):
                    if score <= self.score_percentiles['p1']:
                        fraud_proba[i] = 0.95
                    elif score <= self.score_percentiles['p5']:
                        fraud_proba[i] = 0.85
                    elif score <= self.score_percentiles['p10']:
                        fraud_proba[i] = 0.70
                    elif score <= self.score_percentiles['p25']:
                        fraud_proba[i] = 0.50
                    elif score <= self.score_percentiles['p50']:
                        fraud_proba[i] = 0.30
                    elif score <= self.score_percentiles['p75']:
                        fraud_proba[i] = 0.15
                    elif score <= self.score_percentiles['p90']:
                        fraud_proba[i] = 0.05
                    else:
                        fraud_proba[i] = 0.02
            else:
                # Smart fallback: Use prediction + score-based estimation
                # One-Class SVM: negative scores = anomalies
                # Be more conservative - reduce false positives
                fraud_proba = np.zeros(len(decision_scores))
                for i, (score, pred) in enumerate(zip(decision_scores, predictions)):
                    if pred == -1:  # Anomaly detected
                        # More negative = higher fraud probability
                        # But be conservative - most are moderate risk
                        if score < -1.0:
                            fraud_proba[i] = 0.70  # Reduced from 0.90
                        elif score < -0.5:
                            fraud_proba[i] = 0.55  # Reduced from 0.75
                        elif score < 0.0:
                            fraud_proba[i] = 0.40  # Reduced from 0.60
                        else:
                            fraud_proba[i] = 0.30  # Reduced from 0.45
                    else:  # Normal - should be low risk
                        if score < 0.5:
                            fraud_proba[i] = 0.15  # Reduced from 0.30
                        elif score < 1.0:
                            fraud_proba[i] = 0.08  # Reduced from 0.15
                        else:
                            fraud_proba[i] = 0.03  # Reduced from 0.05
        
        return fraud_proba
    
    def predict(self, X, threshold=0.5):
        """Predict binary fraud labels"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc,
            'confusion_matrix': cm.tolist()
        }
    
    def save(self, path):
        """Save model with calibration parameters"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            'model': self.model,
            'score_min': self.score_min,
            'score_max': self.score_max,
            'score_percentiles': self.score_percentiles
        }
        joblib.dump(save_data, path)
    
    def load(self, path):
        """Load model with calibration parameters"""
        data = joblib.load(path)
        if isinstance(data, dict):
            self.model = data['model']
            self.score_min = data.get('score_min')
            self.score_max = data.get('score_max')
            self.score_percentiles = data.get('score_percentiles')
        else:
            # Backward compatibility: old format without calibration
            self.model = data
            self.score_min = None
            self.score_max = None
            self.score_percentiles = None
        return self


class AutoencoderModel:
    """Autoencoder for anomaly detection"""
    
    def __init__(self, config):
        self.config = config
        self.encoding_dim = config['model']['autoencoder']['encoding_dim']
        self.model = None
        self.name = "Autoencoder"
        self.input_dim = None
        # Calibration parameters for probability conversion
        self.error_min = None
        self.error_max = None
        self.error_percentiles = None
    
    def _build_model(self, input_dim):
        """Build autoencoder architecture"""
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config['model']['autoencoder']['learning_rate']
            ),
            loss='mse'
        )
        
        return autoencoder
    
    def train(self, X_train, y_train=None):
        """Train the autoencoder on normal samples only"""
        self.input_dim = X_train.shape[1]
        self.model = self._build_model(self.input_dim)
        
        # Use only normal samples for training
        if y_train is not None:
            normal_mask = y_train == 0
            X_normal = X_train[normal_mask]
        else:
            X_normal = X_train
        
        history = self.model.fit(
            X_normal, X_normal,
            epochs=self.config['model']['autoencoder']['epochs'],
            batch_size=self.config['model']['autoencoder']['batch_size'],
            validation_split=self.config['model']['autoencoder']['validation_split'],
            verbose=0
        )
        
        # Calculate calibration parameters from training data
        # Use batch prediction for efficiency on large datasets
        X_pred = self.model.predict(X_normal, verbose=0, batch_size=256)
        train_errors = np.mean(np.square(X_normal - X_pred), axis=1)
        self.error_min = np.min(train_errors)
        self.error_max = np.max(train_errors)
        # Store percentiles for better probability estimation
        self.error_percentiles = {
            'p1': np.percentile(train_errors, 1),
            'p5': np.percentile(train_errors, 5),
            'p10': np.percentile(train_errors, 10),
            'p25': np.percentile(train_errors, 25),
            'p50': np.percentile(train_errors, 50),
            'p75': np.percentile(train_errors, 75),
            'p90': np.percentile(train_errors, 90),
            'p95': np.percentile(train_errors, 95),
            'p99': np.percentile(train_errors, 99)
        }
        
        return history
    
    def predict_proba(self, X):
        """Predict fraud probability based on reconstruction error using calibrated parameters"""
        # Ensure X is a numpy array with correct dtype
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Ensure float32 for TensorFlow compatibility
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Optimize for single samples: use direct call instead of predict()
        # This is much faster for single-sample inference (common in real-time predictions)
        try:
            if len(X) == 1:
                # Direct call is faster for single samples (avoids predict() overhead)
                X_pred = self.model(X, training=False)
                # Convert tensor to numpy array
                if hasattr(X_pred, 'numpy'):
                    X_pred = X_pred.numpy()
                elif hasattr(X_pred, '__array__'):
                    X_pred = np.array(X_pred)
            elif len(X) < 32:
                # For small batches, use direct call
                X_pred = self.model(X, training=False)
                if hasattr(X_pred, 'numpy'):
                    X_pred = X_pred.numpy()
                elif hasattr(X_pred, '__array__'):
                    X_pred = np.array(X_pred)
            else:
                # For larger batches, use predict with batch_size for efficiency
                X_pred = self.model.predict(X, verbose=0, batch_size=256)
        except Exception as e:
            # Fallback to predict() if direct call fails
            X_pred = self.model.predict(X, verbose=0, batch_size=256)
        
        # Calculate reconstruction error
        mse = np.mean(np.square(X - X_pred), axis=1)
        
        # Use calibration parameters if available
        if self.error_min is not None and self.error_max is not None:
            error_range = self.error_max - self.error_min
            if error_range > 0:
                fraud_proba = (mse - self.error_min) / error_range
            else:
                fraud_proba = np.zeros(len(mse))
        else:
            # Fallback: use percentile-based estimation if available
            if self.error_percentiles is not None:
                fraud_proba = np.zeros(len(mse))
                for i, error in enumerate(mse):
                    if error >= self.error_percentiles['p99']:
                        fraud_proba[i] = 0.95
                    elif error >= self.error_percentiles['p95']:
                        fraud_proba[i] = 0.85
                    elif error >= self.error_percentiles['p90']:
                        fraud_proba[i] = 0.70
                    elif error >= self.error_percentiles['p75']:
                        fraud_proba[i] = 0.50
                    elif error >= self.error_percentiles['p50']:
                        fraud_proba[i] = 0.30
                    elif error >= self.error_percentiles['p25']:
                        fraud_proba[i] = 0.15
                    elif error >= self.error_percentiles['p10']:
                        fraud_proba[i] = 0.05
                    else:
                        fraud_proba[i] = 0.02
            else:
                # Smart fallback: Use error magnitude for probability estimation
                # Higher reconstruction error = higher fraud probability
                # Typical errors: 0.01-0.1 (normal), 0.1-1.0 (suspicious), >1.0 (fraud)
                # Be more conservative to reduce false positives
                fraud_proba = np.zeros(len(mse))
                for i, error in enumerate(mse):
                    if error > 1.0:
                        fraud_proba[i] = 0.70  # Reduced from 0.90
                    elif error > 0.5:
                        fraud_proba[i] = 0.55  # Reduced from 0.75
                    elif error > 0.2:
                        fraud_proba[i] = 0.40  # Reduced from 0.60
                    elif error > 0.1:
                        fraud_proba[i] = 0.30  # Reduced from 0.40
                    elif error > 0.05:
                        fraud_proba[i] = 0.20  # Reduced from 0.25
                    elif error > 0.02:
                        fraud_proba[i] = 0.10  # Reduced from 0.15
                    else:
                        fraud_proba[i] = 0.05  # Keep low for normal
        
        return fraud_proba
    
    def predict(self, X, threshold=0.5):
        """Predict binary fraud labels"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc,
            'confusion_matrix': cm.tolist()
        }
    
    def save(self, path):
        """Save model with calibration parameters"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save Keras model
        self.model.save(path)
        # Save calibration parameters separately
        calib_path = path.replace('.h5', '_calibration.joblib')
        calib_data = {
            'error_min': self.error_min,
            'error_max': self.error_max,
            'error_percentiles': self.error_percentiles
        }
        joblib.dump(calib_data, calib_path)
    
    def load(self, path):
        """Load model with calibration parameters"""
        try:
            # Load with compile=False and safe_mode=False to avoid Keras 2.x -> 3.x compatibility issues
            # This skips loading the optimizer and metrics which have changed in Keras 3.x
            # safe_mode=False allows loading models saved with older Keras versions
            self.model = keras.models.load_model(path, compile=False, safe_mode=False)
            
            # Recompile with compatible settings for Keras 3.x
            # We don't need metrics for inference, just the loss function
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=self.config['model']['autoencoder']['learning_rate']
                ),
                loss='mse'
            )
        except Exception as e:
            # If loading fails, provide a more informative error
            raise Exception(
                f"Failed to load Keras model from {path}. "
                f"This may be due to Keras version incompatibility. "
                f"Model was likely saved with Keras 2.x but you're using Keras 3.x. "
                f"Error: {str(e)}. "
                f"Consider retraining the model with the current Keras version."
            )
        
        # Infer input_dim from loaded model
        if hasattr(self.model, 'input_shape') and self.model.input_shape:
            self.input_dim = self.model.input_shape[1]
        else:
            # Fallback: try to get from input layer
            try:
                self.input_dim = self.model.input.shape[1]
            except:
                raise Exception("Could not determine input dimension from loaded model")
        
        # Load calibration parameters
        calib_path = path.replace('.h5', '_calibration.joblib')
        try:
            calib_data = joblib.load(calib_path)
            self.error_min = calib_data.get('error_min')
            self.error_max = calib_data.get('error_max')
            self.error_percentiles = calib_data.get('error_percentiles')
        except:
            # Backward compatibility: no calibration data
            self.error_min = None
            self.error_max = None
            self.error_percentiles = None
        return self

