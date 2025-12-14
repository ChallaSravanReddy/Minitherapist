# Advanced Emotion Classification Training Results

## Dataset Information
- **Source**: `Dataset/emotion-emotion_69k.csv`
- **Total Records**: 64,636
- **Processed Records**: ~10,000 (sampled for faster training)
- **Text Column**: Situation
- **Label Column**: emotion
- **Number of Emotion Classes**: 32

### Emotion Distribution (Top 15)
- joyful: 1,953
- surprised: 1,904
- angry: 2,247
- annoyed: 2,247
- sad: 2,213
- lonely: 2,213
- afraid: 2,106
- grateful: 2,091
- terrified: 2,074
- guilty: 2,053
- furious: 2,045
- disgusted: 2,044
- confident: 2,037
- anxious: 2,037
- anticipating: 2,026

## Data Split
- **Training Set**: 6,999 samples (70%)
- **Validation Set**: 1,000 samples (10%)
- **Test Set**: 2,000 samples (20%)

## Models Trained

### 1. Logistic Regression (TF-IDF)
**Features**: TF-IDF Vectorization with unigrams and bigrams
- **Accuracy**: 0.4695 (46.95%)
- **Precision**: 0.4672
- **Recall**: 0.4695
- **F1-Score**: 0.4628

### 2. Random Forest (TF-IDF)
**Features**: TF-IDF Vectorization with unigrams and bigrams
- **Accuracy**: 0.5340 (53.40%)
- **Precision**: 0.5409
- **Recall**: 0.5340
- **F1-Score**: 0.5312
- **Status**: ✓ Best Traditional Model

### 3. DistilBERT (Transformer-based)
**Architecture**: Hugging Face DistilBERT with 32 emotion classes
- **Training Epochs**: 2
- **Batch Size**: 32
- **Max Sequence Length**: 128
- **Optimizer**: AdamW (lr=2e-5)
- **Status**: Training in progress...

## Advanced NLP Features Used

### 1. **TF-IDF Vectorization**
   - Max features: 2,000
   - N-gram range: (1, 2) - unigrams and bigrams
   - Min document frequency: 2
   - Max document frequency: 0.8
   - Sublinear TF scaling enabled

### 2. **Transformer Models**
   - **DistilBERT**: Lightweight BERT variant
   - Pre-trained on English text
   - Fine-tuned for emotion classification
   - Contextual embeddings capture semantic meaning

### 3. **Ensemble Approach**
   - Multiple models for comparison
   - Logistic Regression for baseline
   - Random Forest for non-linear patterns
   - Transformer for deep contextual understanding

### 4. **Data Preprocessing**
   - Text normalization (lowercase, whitespace)
   - Removal of corrupted entries
   - Stratified train-test split
   - Class imbalance handling

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 46.95% | 0.4672 | 0.4695 | 0.4628 |
| Random Forest | 53.40% | 0.5409 | 0.5340 | 0.5312 |
| DistilBERT | *Training* | *Training* | *Training* | *Training* |

## Key Insights

1. **Random Forest outperforms Logistic Regression** by ~6.5% accuracy, suggesting non-linear patterns in emotion text
2. **32 emotion classes** is a challenging multi-class problem
3. **DistilBERT** is expected to provide better performance due to:
   - Contextual word embeddings
   - Attention mechanisms
   - Pre-trained language understanding

## Files Generated

- `models/logistic_regression.pkl` - Trained LR model
- `models/random_forest.pkl` - Trained RF model
- `models/distilbert.pkl` - Trained transformer model (when complete)
- `models/label_encoder.pkl` - Emotion label encoder
- `models/training_report.json` - Detailed metrics in JSON format

## Next Steps

1. ✓ Train TF-IDF based models
2. ⏳ Train DistilBERT transformer model
3. Generate comprehensive evaluation report
4. Integrate best model into chatbot
5. Test on real user inputs

## System Configuration

- **Device**: CPU (GPU available if CUDA detected)
- **Python Version**: 3.12
- **Key Libraries**:
  - scikit-learn 1.3.2
  - transformers 4.57.3
  - torch 2.6.0
  - pandas 2.1.3
