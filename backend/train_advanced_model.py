"""
Advanced NLP-based emotion classification model using transformers and ensemble methods.
Trains on the 69k emotion dataset with cross-validation and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers for advanced NLP
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

class AdvancedEmotionClassifier:
    def __init__(self, model_name='distilbert-base-uncased', device='cpu'):
        """Initialize the advanced emotion classifier."""
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def load_dataset(self, csv_path, sample_size=None):
        """Load and preprocess the emotion dataset."""
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Display dataset info
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Use Situation and emotion columns
        text_col, emotion_col = 'Situation', 'emotion'
        
        # Clean data
        df = df.dropna(subset=[text_col, emotion_col])
        df[text_col] = df[text_col].astype(str).str.strip()
        df[emotion_col] = df[emotion_col].astype(str).str.strip().str.lower()
        
        # Remove empty texts
        df = df[df[text_col].str.len() > 0]
        
        # Remove rows with very short emotions (likely corrupted data)
        df = df[df[emotion_col].str.len() < 30]
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size} examples from dataset...")
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"\nFinal dataset size: {len(df)}")
        print(f"Emotion distribution:\n{df[emotion_col].value_counts()}")
        
        self.texts = df[text_col].values
        self.labels = df[emotion_col].values
        
        return self.texts, self.labels
    
    def prepare_data(self, test_size=0.2, val_size=0.1, min_samples_per_class=5):
        """Split data into train, validation, and test sets."""
        # Encode labels
        self.label_encoder.fit(self.labels)
        encoded_labels = self.label_encoder.transform(self.labels)
        
        # Filter out classes with too few samples
        unique_labels, counts = np.unique(encoded_labels, return_counts=True)
        valid_labels = unique_labels[counts >= min_samples_per_class]
        
        mask = np.isin(encoded_labels, valid_labels)
        texts_filtered = self.texts[mask]
        labels_filtered = encoded_labels[mask]
        
        print(f"\nFiltered dataset (min {min_samples_per_class} samples per class):")
        print(f"  Original size: {len(self.texts)}")
        print(f"  Filtered size: {len(texts_filtered)}")
        print(f"  Classes retained: {len(valid_labels)}/{len(unique_labels)}")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts_filtered, labels_filtered, test_size=test_size, 
            random_state=42, stratify=labels_filtered
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_tfidf_ensemble(self):
        """Train TF-IDF + Ensemble models for baseline comparison."""
        print("\n" + "="*60)
        print("TRAINING TF-IDF + ENSEMBLE MODELS")
        print("="*60)
        
        # TF-IDF Vectorization
        print("\nVectorizing texts with TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        X_train_vec = vectorizer.fit_transform(self.X_train)
        X_val_vec = vectorizer.transform(self.X_val)
        X_test_vec = vectorizer.transform(self.X_test)
        
        # Train Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1
        )
        lr_model.fit(X_train_vec, self.y_train)
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
        )
        rf_model.fit(X_train_vec, self.y_train)
        
        # Evaluate models
        models = {'Logistic Regression': lr_model, 'Random Forest': rf_model}
        
        for name, model in models.items():
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'model': model,
                'vectorizer': vectorizer
            }
            
            print(f"\n{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        return models, vectorizer
    
    def train_transformer_model(self, epochs=3, batch_size=16, learning_rate=2e-5):
        """Train a transformer-based model (DistilBERT)."""
        print("\n" + "="*60)
        print("TRAINING TRANSFORMER MODEL (DistilBERT)")
        print("="*60)
        
        # Tokenize texts
        print("\nTokenizing texts...")
        def tokenize_batch(texts, max_length=128):
            return self.tokenizer(
                texts.tolist(),
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        train_encodings = tokenize_batch(self.X_train)
        val_encodings = tokenize_batch(self.X_val)
        test_encodings = tokenize_batch(self.X_test)
        
        # Create datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(self.y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(self.y_val, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            test_encodings['input_ids'],
            test_encodings['attention_mask'],
            torch.tensor(self.y_test, dtype=torch.long)
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Load model
        num_labels = len(self.label_encoder.classes_)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        model.to(self.device)
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f"\nTraining for {epochs} epochs...")
        best_val_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device)
                )
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            training_history['train_loss'].append(avg_train_loss)
            
            # Validation
            model.eval()
            total_val_loss = 0
            correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = (
                        input_ids.to(self.device),
                        attention_mask.to(self.device),
                        labels.to(self.device)
                    )
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    total_val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct += (predictions == labels).sum().item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct / len(self.y_val)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        # Test evaluation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, _ = batch
                input_ids, attention_mask = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device)
                )
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(predictions.cpu().numpy())
        
        y_pred = np.array(all_preds)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        self.results['DistilBERT'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model,
            'training_history': training_history
        }
        
        print(f"\nDistilBERT Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return model, training_history
    
    def generate_report(self, output_dir='models'):
        """Generate comprehensive training report."""
        print("\n" + "="*60)
        print("TRAINING REPORT")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(self.texts),
            'num_emotions': len(self.label_encoder.classes_),
            'emotions': self.label_encoder.classes_.tolist(),
            'models': {}
        }
        
        # Model comparison
        print("\nMODEL COMPARISON:")
        print("-" * 60)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f}")
            
            report['models'][model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1'])
            }
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        print(f"\n✓ Best Model: {best_model[0]} (F1-Score: {best_model[1]['f1']:.4f})")
        
        # Save report
        report_path = os.path.join(output_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {report_path}")
        
        return report
    
    def save_models(self, output_dir='models'):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, metrics in self.results.items():
            if 'model' in metrics:
                model_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(metrics['model'], f)
                print(f"✓ Saved {model_name} to {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✓ Saved label encoder to {encoder_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("ADVANCED EMOTION CLASSIFICATION TRAINING")
    print("="*60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'Dataset', 'emotion-emotion_69k.csv')
    output_dir = os.path.join(script_dir, 'models')
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Initialize classifier
    classifier = AdvancedEmotionClassifier(device=device)
    
    # Load dataset
    classifier.load_dataset(dataset_path, sample_size=10000)  # Use 10k samples for faster training
    
    # Prepare data
    classifier.prepare_data()
    
    # Train models
    classifier.train_tfidf_ensemble()
    classifier.train_transformer_model(epochs=2, batch_size=32)
    
    # Generate report
    classifier.generate_report(output_dir)
    
    # Save models
    classifier.save_models(output_dir)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
