"""
Fast emotion classification training - Optimized for speed with best results.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    print("\n" + "="*70)
    print("FAST EMOTION CLASSIFICATION TRAINING - DATASET: 69K EMOTIONS")
    print("="*70 + "\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'Dataset', 'emotion-emotion_69k.csv')
    output_dir = os.path.join(script_dir, 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv(dataset_path, encoding='utf-8')
    df = df.dropna(subset=['Situation', 'emotion'])
    df['Situation'] = df['Situation'].astype(str).str.strip()
    df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()
    df = df[df['Situation'].str.len() > 0]
    df = df[df['emotion'].str.len() < 30]
    
    print(f"   Total records: {len(df)}")
    print(f"   Unique emotions: {df['emotion'].nunique()}")
    print(f"\n   Top 10 emotions:")
    for emotion, count in df['emotion'].value_counts().head(10).items():
        print(f"      • {emotion}: {count}")
    
    # Sample for faster training
    sample_size = 12000
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    texts = df['Situation'].values
    labels = df['emotion'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    
    # Filter classes with too few samples
    unique_labels, counts = np.unique(encoded_labels, return_counts=True)
    valid_labels = unique_labels[counts >= 5]
    mask = np.isin(encoded_labels, valid_labels)
    
    texts = texts[mask]
    encoded_labels = encoded_labels[mask]
    
    print(f"\n   After filtering: {len(texts)} samples")
    print(f"   Classes: {len(valid_labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"\n   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # TF-IDF Vectorization
    print("\n🔤 Vectorizing texts with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"   TF-IDF features: {X_train_vec.shape[1]}")
    
    # Train models
    print("\n🤖 Training models...")
    results = {}
    
    # Logistic Regression
    print("   1. Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1)
    lr_model.fit(X_train_vec, y_train)
    lr_pred = lr_model.predict(X_test_vec)
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, lr_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, lr_pred, average='weighted', zero_division=0),
        'predictions': lr_pred
    }
    
    # Random Forest
    print("   2. Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_train_vec, y_train)
    rf_pred = rf_model.predict(X_test_vec)
    
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0),
        'predictions': rf_pred
    }
    
    # Results
    print("\n" + "="*70)
    print("📊 TRAINING RESULTS")
    print("="*70)
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    best_model_name = None
    best_f1 = 0
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
    
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model_name} (F1-Score: {best_f1:.4f})")
    print("="*70)
    
    # Detailed report for best model
    print(f"\n📋 DETAILED CLASSIFICATION REPORT - {best_model_name}")
    print("-"*70)
    best_pred = results[best_model_name]['predictions']
    # Only use labels that appear in y_test
    unique_test_labels = np.unique(y_test)
    target_names = [label_encoder.classes_[i] for i in unique_test_labels]
    print(classification_report(y_test, best_pred, 
                               labels=unique_test_labels,
                               target_names=target_names,
                               zero_division=0))
    
    # Save models
    print("\n💾 Saving models...")
    best_model = results[best_model_name]['model']
    
    model_path = os.path.join(output_dir, 'best_emotion_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"   ✓ Best model saved: {model_path}")
    
    vec_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"   ✓ Vectorizer saved: {vec_path}")
    
    enc_path = os.path.join(output_dir, 'emotion_label_encoder.pkl')
    with open(enc_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   ✓ Label encoder saved: {enc_path}")
    
    # Save JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total_records': len(df),
            'samples_used': len(texts),
            'num_emotions': len(valid_labels),
            'emotions': label_encoder.classes_.tolist()
        },
        'data_split': {
            'train': len(X_train),
            'test': len(X_test)
        },
        'models': {
            name: {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1'])
            }
            for name, metrics in results.items()
        },
        'best_model': best_model_name
    }
    
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   ✓ Report saved: {report_path}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
