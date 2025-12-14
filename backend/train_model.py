import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import re
import numpy as np

def preprocess_text(text):
    """Preprocess text for training."""
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^a-z0-9\s\'\-]', '', text)
    return text

def load_training_data(data_dir):
    """Load training data from processed JSON files."""
    data = []
    
    # Load merged dataset
    merged_path = os.path.join(data_dir, 'merged_training_data.json')
    if os.path.exists(merged_path):
        print(f"Loading merged dataset from {merged_path}...")
        with open(merged_path, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
            data.extend(merged_data)
            print(f"  Loaded {len(merged_data)} examples from merged dataset")
    
    # Also load original small dataset if exists (high quality)
    original_path = os.path.join(data_dir, 'emotions_training.json')
    if os.path.exists(original_path):
        print(f"Loading original dataset from {original_path}...")
        with open(original_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            data.extend(original_data)
            print(f"  Loaded {len(original_data)} examples from original dataset")
    
    if not data:
        raise ValueError("No training data found! Please run process_datasets.py first.")
    
    texts = [item['text'] for item in data]
    labels = [item['label'] if 'label' in item else item.get('emotion', 'neutral') for item in data]
    
    return texts, labels

def train_model(data_dir, model_output_path):
    """Train the emotion classification model with advanced features."""
    print("\n" + "="*60)
    print("EMOTION CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    print("\n[1/6] Loading training data...")
    texts, labels = load_training_data(data_dir)
    
    print(f"\nLoaded {len(texts)} training examples")
    
    # Print emotion distribution
    from collections import Counter
    emotion_counts = Counter(labels)
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")
    
    # Preprocess texts
    print("\n[2/6] Preprocessing texts...")
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Split data with stratification
    print("\n[3/6] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"  Training set: {len(X_train)} examples")
    print(f"  Test set: {len(X_test)} examples")
    
    # Create TF-IDF vectorizer with advanced features
    print("\n[4/6] Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=2000,  # Increased from 1000
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,  # Use sublinear tf scaling
        analyzer='word'
    )
    
    # Fit and transform training data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"  Feature matrix shape: {X_train_vec.shape}")
    
    # Handle class imbalance with SMOTE (if needed)
    print("\n[5/6] Training model...")
    try:
        # Check if we need SMOTE
        min_class_count = min(Counter(y_train).values())
        if min_class_count < 10:
            print("  Skipping SMOTE (insufficient samples in some classes)")
            X_train_resampled = X_train_vec
            y_train_resampled = y_train
        else:
            print("  Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)
            print(f"  Resampled to {len(y_train_resampled)} examples")
    except Exception as e:
        print(f"  SMOTE failed ({e}), using original data")
        X_train_resampled = X_train_vec
        y_train_resampled = y_train
    
    # Train Logistic Regression with hyperparameter tuning
    print("\n  Training Logistic Regression with GridSearch...")
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000],
        'class_weight': ['balanced']
    }
    
    lr = LogisticRegression(random_state=42, solver='lbfgs')
    grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    model = grid_search.best_estimator_
    print(f"  Best parameters: {grid_search.best_params_}")
    
    # Evaluate model
    print("\n[6/6] Evaluating model...")
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Cross-validation on original training set
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Test set evaluation
    y_pred = model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {test_accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model and vectorizer
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'metadata': {
            'training_date': pd.Timestamp.now().isoformat() if 'pd' in dir() else 'unknown',
            'training_samples': len(texts),
            'test_accuracy': float(test_accuracy),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'emotion_labels': sorted(list(set(labels))),
            'best_params': grid_search.best_params_
        }
    }
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to {model_output_path}")
    print(f"✓ Test accuracy: {test_accuracy:.1%}")
    print(f"✓ CV accuracy: {cv_scores.mean():.1%}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, vectorizer

if __name__ == '__main__':
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'training')
    model_path = os.path.join(script_dir, 'models', 'trained_model.pkl')
    
    # Train model
    try:
        import pandas as pd  # For timestamp
    except:
        pass
    
    train_model(data_dir, model_path)
