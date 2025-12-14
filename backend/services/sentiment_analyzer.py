from textblob import TextBlob
from typing import Dict, Tuple
import re

class SentimentAnalyzer:
    """Advanced sentiment analysis service using TextBlob."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.emotion_intensity_map = {
            'sad': -0.5,
            'stressed': -0.4,
            'angry': -0.6,
            'lonely': -0.5,
            'confused': -0.2,
            'overwhelmed': -0.6,
            'happy': 0.6,
            'worried': -0.4,
            'neutral': 0.0
        }
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            blob = TextBlob(text)
            
            # Get polarity (-1 to 1) and subjectivity (0 to 1)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment category
            if polarity > 0.3:
                sentiment = 'positive'
            elif polarity < -0.3:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Calculate intensity (0 to 1)
            intensity = abs(polarity)
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'intensity': intensity
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'sentiment': 'neutral',
                'intensity': 0.0
            }
    
    def get_emotion_intensity(self, emotion: str, text: str) -> float:
        """Calculate emotion intensity based on sentiment and emotion type.
        
        Args:
            emotion: Detected emotion
            text: User message
            
        Returns:
            Intensity score (0 to 1)
        """
        sentiment = self.analyze(text)
        base_intensity = self.emotion_intensity_map.get(emotion, 0.0)
        
        # Adjust intensity based on sentiment polarity
        if base_intensity < 0:  # Negative emotions
            # More negative polarity = higher intensity
            intensity = min(1.0, abs(base_intensity) + abs(min(0, sentiment['polarity'])))
        else:  # Positive emotions
            # More positive polarity = higher intensity
            intensity = min(1.0, base_intensity + max(0, sentiment['polarity']))
        
        return intensity
    
    def detect_aspects(self, text: str) -> Dict:
        """Detect what the user is talking about (aspect-based sentiment).
        
        Args:
            text: User message
            
        Returns:
            Dictionary with detected aspects
        """
        text_lower = text.lower()
        
        aspects = {
            'work': ['job', 'work', 'career', 'business', 'office', 'boss', 'colleague'],
            'family': ['family', 'parent', 'mother', 'father', 'mom', 'dad', 'sibling', 'brother', 'sister'],
            'relationships': ['relationship', 'partner', 'boyfriend', 'girlfriend', 'spouse', 'friend'],
            'health': ['health', 'sick', 'pain', 'doctor', 'hospital', 'illness'],
            'finance': ['money', 'financial', 'debt', 'bills', 'salary', 'income'],
            'education': ['school', 'college', 'university', 'study', 'exam', 'grade'],
            'self': ['myself', 'i feel', 'i am', 'i\'m', 'me', 'my life']
        }
        
        detected_aspects = []
        for aspect, keywords in aspects.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_aspects.append(aspect)
        
        return {
            'aspects': detected_aspects,
            'primary_aspect': detected_aspects[0] if detected_aspects else 'general'
        }
    
    def get_sentiment_trend(self, messages: list) -> Dict:
        """Analyze sentiment trend over conversation.
        
        Args:
            messages: List of recent messages
            
        Returns:
            Dictionary with trend information
        """
        if not messages:
            return {'trend': 'stable', 'direction': 'neutral'}
        
        # Analyze last 5 messages
        recent_messages = messages[-5:]
        polarities = []
        
        for msg in recent_messages:
            sentiment = self.analyze(msg)
            polarities.append(sentiment['polarity'])
        
        # Calculate trend
        if len(polarities) >= 2:
            # Compare first half to second half
            mid = len(polarities) // 2
            first_half_avg = sum(polarities[:mid]) / mid if mid > 0 else 0
            second_half_avg = sum(polarities[mid:]) / (len(polarities) - mid)
            
            diff = second_half_avg - first_half_avg
            
            if diff > 0.2:
                trend = 'improving'
                direction = 'positive'
            elif diff < -0.2:
                trend = 'declining'
                direction = 'negative'
            else:
                trend = 'stable'
                direction = 'neutral' if abs(second_half_avg) < 0.2 else ('positive' if second_half_avg > 0 else 'negative')
        else:
            trend = 'stable'
            direction = 'neutral'
        
        return {
            'trend': trend,
            'direction': direction,
            'recent_polarity': polarities[-1] if polarities else 0.0
        }
