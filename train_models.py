"""
Training Script for ML Assignment 2 - Earthquake Tsunami Prediction
Trains 6 different classification models and saves results
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Calculate all evaluation metrics"""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def main():
    print("="*80)
    print("ML ASSIGNMENT 2 - MODEL TRAINING")
    print("Earthquake Tsunami Prediction - Binary Classification")
    print("="*80)
    print()
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Load data
    print("📁 Loading training dataset...")
    train_data = pd.read_csv('earthquake_tsunami_train.csv')
    print(f"✅ Loaded {len(train_data)} training samples with {len(train_data.columns)} features")
    print()
    
    # Prepare training data
    print("🔧 Preparing training data...")
    X_train = train_data.drop('tsunami', axis=1)
    y_train = train_data['tsunami'].values
    
    # Load test data
    print("📁 Loading test dataset...")
    test_data = pd.read_csv('earthquake_tsunami_test.csv')
    print(f"✅ Loaded {len(test_data)} test samples")
    
    X_test = test_data.drop('tsunami', axis=1)
    y_test = test_data['tsunami'].values
    
    print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print()
    
    # Scale features
    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved")
    print()
    
    # Train models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes': GaussianNB(),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    all_results = []
    
    print("🤖 Training models...")
    print("-"*80)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name.replace('_', ' ').title()}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, name.replace('_', ' ').title())
        all_results.append(metrics)
        
        # Save model
        model_path = f'models/{name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Trained and saved {name}")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC Score']:.4f}, F1: {metrics['F1 Score']:.4f}")
    
    print()
    print("-"*80)
    print("✅ All models trained successfully!")
    print()
    
    # Save results
    print("💾 Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.round(4)
    results_df.to_csv('models/model_results.csv', index=False)
    print("✅ Results saved to models/model_results.csv")
    print()
    
    # Display results
    print("="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    print()
    
    # Best models
    print("🏆 BEST MODEL BY METRIC")
    print("-"*80)
    for metric in ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"{metric:20s}: {best_model:25s} ({best_score:.4f})")
    print("-"*80)
    print()
    
    print("✅ Training complete! You can now run the Streamlit app:")
    print("   streamlit run app.py")
    print()

if __name__ == "__main__":
    main()
