"""
Quick training results generator - Uses only TF-IDF models for immediate results.
Provides comprehensive evaluation and metrics.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

class QuickEmotionTrainer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def load_and_prepare(self, csv_path, sample_size=15000):
        """Load and prepare dataset."""
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Use Situation and emotion columns
        df = df.dropna(subset=['Situation', 'emotion'])
        df['Situation'] = df['Situation'].astype(str).str.strip()
        df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()
        df = df[df['Situation'].str.len() > 0]
        df = df[df['emotion'].str.len() < 30]
        
        # Sample
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"Dataset size: {len(df)}")
        print(f"Emotion distribution:\n{df['emotion'].value_counts()}\n")
        
        texts = df['Situation'].values
        labels = df['emotion'].values
        
        # Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Filter classes with too few samples
        unique_labels, counts = np.unique(encoded_labels, return_counts=True)
        valid_labels = unique_labels[counts >= 5]
        mask = np.isin(encoded_labels, valid_labels)
        
        texts = texts[mask]
        encoded_labels = encoded_labels[mask]
        
        print(f"After filtering: {len(texts)} samples, {len(valid_labels)} classes")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models."""
        print("="*60)
        print("VECTORIZING TEXTS WITH TF-IDF")
        print("="*60)
        
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print(f"TF-IDF shape: {X_train_vec.shape}\n")
        
        # Train models
        models_config = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Linear SVM': LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
        }
        
        print("="*60)
        print("TRAINING MODELS")
        print("="*60 + "\n")
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(X_train_vec, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_vec)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred
            }
            
            print(f"  ✓ Accuracy: {accuracy:.4f}")
            print(f"  ✓ Precision: {precision:.4f}")
            print(f"  ✓ Recall: {recall:.4f}")
            print(f"  ✓ F1-Score: {f1:.4f}\n")
        
        return vectorizer, X_test_vec, y_test
    
    def generate_report(self, X_test_vec, y_test, output_dir='models'):
        """Generate comprehensive report."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("TRAINING RESULTS SUMMARY")
        print("="*60 + "\n")
        
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        best_model_name = None
        best_f1 = 0
        
        for name, metrics in self.results.items():
            print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
            
            report['models'][name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1'])
            }
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model_name = name
        
        print("\n" + "="*60)
        print(f"✓ BEST MODEL: {best_model_name} (F1-Score: {best_f1:.4f})")
        print("="*60 + "\n")
        
        # Detailed classification report for best model
        print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
        print("-"*60)
        best_pred = self.results[best_model_name]['y_pred']
        print(classification_report(y_test, best_pred, 
                                   target_names=self.label_encoder.classes_,
                                   zero_division=0))
        
        # Save report
        report_path = os.path.join(output_dir, 'quick_training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {report_path}")
        
        return report, best_model_name
    
    def save_models(self, vectorizer, output_dir='models'):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1'])[0]
        best_model = self.results[best_model_name]['model']
        
        model_path = os.path.join(output_dir, 'best_emotion_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✓ Saved best model ({best_model_name}) to {model_path}")
        
        # Save vectorizer
        vec_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
        with open(vec_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"✓ Saved vectorizer to {vec_path}")
        
        # Save label encoder
        enc_path = os.path.join(output_dir, 'emotion_label_encoder.pkl')
        with open(enc_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Saved label encoder to {enc_path}")


def main():
    print("\n" + "="*60)
    print("QUICK EMOTION CLASSIFICATION TRAINING")
    print("="*60 + "\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'Dataset', 'emotion-emotion_69k.csv')
    output_dir = os.path.join(script_dir, 'models')
    
    trainer = QuickEmotionTrainer()
    
    # Load and prepare
    X_train, X_test, y_train, y_test = trainer.load_and_prepare(dataset_path, sample_size=15000)
    
    # Train models
    vectorizer, X_test_vec, y_test = trainer.train_models(X_train, X_test, y_train, y_test)
    
    # Generate report
    trainer.generate_report(X_test_vec, y_test, output_dir)
    
    # Save models
    trainer.save_models(vectorizer, output_dir)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
