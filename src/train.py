"""
Model Training and Evaluation Script
Trains all three models and compares their performance
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.feature_engineering import prepare_data
from src.models import IsolationForestModel, OneClassSVMModel, AutoencoderModel


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_and_evaluate_models(config):
    """Train and evaluate all three models"""
    print("=" * 80)
    print("ANOMALY DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # Prepare data
    print("\n[1/5] Preparing data...")
    data = prepare_data(config)
    print(f"  Training set: {len(data['X_train'])} samples")
    print(f"  Validation set: {len(data['X_val'])} samples")
    print(f"  Test set: {len(data['X_test'])} samples")
    print(f"  Fraud rate in training: {data['y_train'].mean():.2%}")
    
    # Initialize models
    models = {
        'isolation_forest': IsolationForestModel(config),
        'one_class_svm': OneClassSVMModel(config),
        'autoencoder': AutoencoderModel(config)
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n[2-4/5] Training {model.name}...")
        
        # Train
        if model_name == 'one_class_svm' or model_name == 'autoencoder':
            model.train(data['X_train'], data['y_train'])
        else:
            model.train(data['X_train'])
        
        # Evaluate on validation set
        print(f"  Evaluating on validation set...")
        val_results = model.evaluate(data['X_val'], data['y_val'])
        results[f'{model_name}_val'] = val_results
        
        # Evaluate on test set
        print(f"  Evaluating on test set...")
        test_results = model.evaluate(data['X_test'], data['y_test'])
        results[f'{model_name}_test'] = test_results
        
        # Save model
        model_path = f"models/{model_name}_model"
        if model_name == 'autoencoder':
            model.save(f"{model_path}.h5")
        else:
            model.save(f"{model_path}.joblib")
        print(f"  Model saved to {model_path}")
        
        # Print results
        print(f"\n  Validation Results:")
        print(f"    Precision: {val_results['precision']:.4f}")
        print(f"    Recall: {val_results['recall']:.4f}")
        print(f"    F1-Score: {val_results['f1']:.4f}")
        print(f"    AUC-ROC: {val_results['auc_roc']:.4f}")
        
        print(f"\n  Test Results:")
        print(f"    Precision: {test_results['precision']:.4f}")
        print(f"    Recall: {test_results['recall']:.4f}")
        print(f"    F1-Score: {test_results['f1']:.4f}")
        print(f"    AUC-ROC: {test_results['auc_roc']:.4f}")
    
    # Compare models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Isolation Forest': [
            results['isolation_forest_test']['precision'],
            results['isolation_forest_test']['recall'],
            results['isolation_forest_test']['f1'],
            results['isolation_forest_test']['auc_roc']
        ],
        'One-Class SVM': [
            results['one_class_svm_test']['precision'],
            results['one_class_svm_test']['recall'],
            results['one_class_svm_test']['f1'],
            results['one_class_svm_test']['auc_roc']
        ],
        'Autoencoder': [
            results['autoencoder_test']['precision'],
            results['autoencoder_test']['recall'],
            results['autoencoder_test']['f1'],
            results['autoencoder_test']['auc_roc']
        ]
    }, index=['Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
    
    print(comparison.to_string())
    
    # Select best model based on F1-score
    best_model_name = max(
        ['isolation_forest', 'one_class_svm', 'autoencoder'],
        key=lambda x: results[f'{x}_test']['f1']
    )
    
    print(f"\nBest Model: {best_model_name.replace('_', ' ').title()} (F1-Score: {results[f'{best_model_name}_test']['f1']:.4f})")
    
    # Save results
    results_path = 'models/training_results.json'
    Path('models').mkdir(exist_ok=True)
    
    # Convert numpy types to native Python types for JSON
    json_results = {}
    for key, value in results.items():
        json_results[key] = {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in value.items()
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results, best_model_name


if __name__ == "__main__":
    config = load_config()
    results, best_model = train_and_evaluate_models(config)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

