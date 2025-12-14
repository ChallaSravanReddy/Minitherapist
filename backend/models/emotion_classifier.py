import json
import pickle
import os
import re
from typing import Dict, Tuple
from textblob import TextBlob

class EmotionClassifier:
    """ML-based emotion classifier using scikit-learn and spaCy preprocessing."""
    
    def __init__(self, model_path: str = None):
        """Initialize the emotion classifier.
        
        Args:
            model_path: Path to the trained model pickle file
        """
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 'trained_model.pkl'
        )
        self.model = None
        self.vectorizer = None
        self.nlp = None
        self.emotion_labels = [
            'sad', 'stressed', 'angry', 'lonely', 'confused',
            'overwhelmed', 'happy', 'worried', 'neutral'
        ]
        
        # Crisis keywords for safety layer integration
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'self harm', 'hurt myself', 'cut myself', 'no reason to live',
            'better off dead', 'end it all'
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and vectorizer from pickle file."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.vectorizer = model_data['vectorizer']
                print(f"✓ Model loaded from {self.model_path}")
            else:
                print(f"⚠ Model file not found at {self.model_path}. Please train the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text using basic NLP techniques.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s\'\-]', '', text)
        
        return text
    
    def check_crisis(self, text: str) -> bool:
        """Check if the text contains crisis keywords.
        
        Args:
            text: User message
            
        Returns:
            True if crisis keywords detected
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict emotion from text.
        
        Args:
            text: User message
            
        Returns:
            Tuple of (emotion, confidence)
        """
        if not self.model or not self.vectorizer:
            # Fallback to rule-based if model not loaded
            return self._rule_based_prediction(text)
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Vectorize
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predict
            prediction = self.model.predict(text_vector)[0]
            probabilities = self.model.predict_proba(text_vector)[0]
            
            # Get confidence score
            confidence = max(probabilities)
            
            # Enhance with sentiment analysis
            try:
                blob = TextBlob(text)
                sentiment_polarity = blob.sentiment.polarity
                
                # Adjust confidence based on sentiment alignment
                if prediction in ['happy'] and sentiment_polarity > 0.3:
                    confidence = min(1.0, confidence + 0.1)
                elif prediction in ['sad', 'stressed', 'angry', 'lonely', 'overwhelmed', 'worried'] and sentiment_polarity < -0.2:
                    confidence = min(1.0, confidence + 0.1)
            except:
                pass
            
            # If confidence is too low, fall back to rule-based
            if confidence < 0.4:
                return self._rule_based_prediction(text)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._rule_based_prediction(text)
    
    def _rule_based_prediction(self, text: str) -> Tuple[str, float]:
        """Fallback rule-based emotion detection using keywords.
        
        Args:
            text: User message
            
        Returns:
            Tuple of (emotion, confidence)
        """
        text_lower = text.lower()
        
        # Emotion keyword patterns
        emotion_patterns = {
            'sad': ['sad', 'depressed', 'down', 'unhappy', 'miserable', 'heartbroken', 'crying', 'empty'],
            'stressed': ['stressed', 'anxious', 'overwhelm', 'pressure', 'tense', 'panic', 'worry', 'nervous'],
            'angry': ['angry', 'mad', 'furious', 'frustrated', 'irritated', 'annoyed', 'pissed', 'rage'],
            'lonely': ['lonely', 'alone', 'isolated', 'nobody', 'abandoned', 'disconnected', 'invisible'],
            'confused': ['confused', 'lost', 'uncertain', 'don\'t know', 'mixed up', 'bewildered', 'unclear'],
            'overwhelmed': ['overwhelmed', 'too much', 'can\'t handle', 'drowning', 'buried', 'swamped'],
            'happy': ['happy', 'great', 'amazing', 'wonderful', 'blessed', 'grateful', 'joyful', 'excited'],
            'worried': ['worried', 'afraid', 'scared', 'concerned', 'fearful', 'anxious', 'nervous', 'uneasy'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'regular', 'nothing']
        }
        
        # Count matches for each emotion
        emotion_scores = {}
        for emotion, keywords in emotion_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return emotion with highest score
        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            # Normalize confidence (simple heuristic)
            confidence = min(emotion_scores[best_emotion] * 0.3, 0.9)
            return best_emotion, confidence
        
        # Default to neutral
        return 'neutral', 0.5
    
    def get_emotion_context(self, messages: list) -> Dict:
        """Analyze emotion trends from conversation history.
        
        Args:
            messages: List of recent messages
            
        Returns:
            Dictionary with emotion analysis
        """
        if not messages:
            return {'primary_emotion': 'neutral', 'trend': 'stable'}
        
        emotions = []
        for msg in messages[-5:]:  # Last 5 messages
            emotion, _ = self.predict(msg)
            emotions.append(emotion)
        
        # Find most common emotion
        from collections import Counter
        emotion_counts = Counter(emotions)
        primary_emotion = emotion_counts.most_common(1)[0][0]
        
        # Determine trend (simplified)
        if len(emotions) >= 2:
            if emotions[-1] == 'happy' and emotions[-2] != 'happy':
                trend = 'improving'
            elif emotions[-1] in ['sad', 'stressed', 'overwhelmed'] and emotions[-2] not in ['sad', 'stressed', 'overwhelmed']:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'primary_emotion': primary_emotion,
            'trend': trend,
            'recent_emotions': emotions
        }
