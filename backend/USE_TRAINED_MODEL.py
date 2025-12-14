"""
Example: How to use the trained emotion classification model in your chatbot.
This demonstrates loading and using the Random Forest model for emotion detection.
"""

import pickle
import os
from pathlib import Path

class EmotionDetector:
    """Emotion detection using trained Random Forest model."""
    
    def __init__(self, model_dir='models'):
        """Initialize emotion detector with trained models."""
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk."""
        print("Loading trained models...")
        
        # Load Random Forest model
        model_path = os.path.join(self.model_dir, 'best_emotion_model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Loaded emotion model from {model_path}")
        
        # Load TF-IDF vectorizer
        vec_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"✓ Loaded vectorizer from {vec_path}")
        
        # Load label encoder
        enc_path = os.path.join(self.model_dir, 'emotion_label_encoder.pkl')
        with open(enc_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"✓ Loaded label encoder from {enc_path}")
        print()
    
    def detect_emotion(self, text, return_probabilities=False):
        """
        Detect emotion from text.
        
        Args:
            text (str): User input text
            return_probabilities (bool): Return probability scores for all emotions
        
        Returns:
            dict: Contains 'emotion' and optionally 'probabilities'
        """
        # Vectorize text
        text_vec = self.vectorizer.transform([text])
        
        # Predict emotion
        emotion_pred = self.model.predict(text_vec)[0]
        emotion_name = self.label_encoder.inverse_transform([emotion_pred])[0]
        
        result = {
            'emotion': emotion_name,
            'confidence': float(self.model.predict_proba(text_vec).max())
        }
        
        # Optionally return probabilities for all emotions
        if return_probabilities:
            probabilities = self.model.predict_proba(text_vec)[0]
            emotion_probs = {
                emotion: float(prob)
                for emotion, prob in zip(self.label_encoder.classes_, probabilities)
            }
            result['all_emotions'] = emotion_probs
        
        return result
    
    def get_supported_emotions(self):
        """Get list of all supported emotions."""
        return list(self.label_encoder.classes_)


def main():
    """Example usage of the emotion detector."""
    
    print("="*70)
    print("EMOTION DETECTION - TRAINED MODEL USAGE EXAMPLE")
    print("="*70)
    print()
    
    # Initialize detector
    detector = EmotionDetector(model_dir='models')
    
    # Get supported emotions
    emotions = detector.get_supported_emotions()
    print(f"Supported emotions ({len(emotions)}):")
    print(f"  {', '.join(emotions[:10])}...")
    print()
    
    # Example test cases
    test_cases = [
        "I just got promoted at work! I'm so excited!",
        "My dog passed away yesterday. I'm devastated.",
        "I'm not sure what to do about this situation.",
        "I can't believe they did that to me. I'm furious!",
        "I'm looking forward to the weekend.",
        "I feel so grateful for my family and friends.",
        "I'm nervous about the presentation tomorrow.",
        "That was hilarious! I haven't laughed that hard in years.",
    ]
    
    print("="*70)
    print("EMOTION DETECTION RESULTS")
    print("="*70)
    print()
    
    for i, text in enumerate(test_cases, 1):
        print(f"Example {i}:")
        print(f"  Text: \"{text}\"")
        
        # Detect emotion
        result = detector.detect_emotion(text, return_probabilities=False)
        
        print(f"  Detected Emotion: {result['emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print()
    
    # Example with probabilities
    print("="*70)
    print("DETAILED EMOTION PROBABILITIES - EXAMPLE")
    print("="*70)
    print()
    
    example_text = "I'm so happy and excited about this opportunity!"
    print(f"Text: \"{example_text}\"")
    print()
    
    result = detector.detect_emotion(example_text, return_probabilities=True)
    
    print(f"Primary Emotion: {result['emotion'].upper()} ({result['confidence']:.2%})")
    print()
    print("Top 5 Probable Emotions:")
    
    # Sort by probability
    sorted_emotions = sorted(
        result['all_emotions'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for emotion, prob in sorted_emotions[:5]:
        print(f"  • {emotion:<20} {prob:>6.2%}")
    
    print()
    print("="*70)
    print("✅ MODEL READY FOR CHATBOT INTEGRATION")
    print("="*70)


# Integration example for Flask/FastAPI
def emotion_api_example():
    """Example of how to integrate with a web API."""
    
    # This would be in your Flask app
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    detector = EmotionDetector(model_dir='models')
    
    @app.route('/detect-emotion', methods=['POST'])
    def detect_emotion_endpoint():
        """API endpoint for emotion detection."""
        data = request.json
        user_text = data.get('text', '')
        
        if not user_text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = detector.detect_emotion(user_text, return_probabilities=True)
        
        return jsonify({
            'text': user_text,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'all_emotions': result['all_emotions']
        })
    
    return app


# Integration example for chatbot response
def chatbot_response_example():
    """Example of how to use emotion detection in chatbot responses."""
    
    detector = EmotionDetector(model_dir='models')
    
    # Emotion-specific responses
    emotion_responses = {
        'happy': "I'm so glad you're feeling happy! That's wonderful! 😊",
        'sad': "I'm sorry you're feeling sad. I'm here to listen and help. 💙",
        'angry': "I understand you're frustrated. Let's talk about what's bothering you.",
        'anxious': "It's normal to feel anxious. Take a deep breath. You've got this! 💪",
        'grateful': "It's beautiful that you're feeling grateful. Gratitude is powerful! 🙏",
        'excited': "Your excitement is contagious! Tell me more about it! 🎉",
        'lonely': "I'm here for you. You're not alone. Let's talk. 💬",
        'confident': "That confidence looks great on you! Keep it up! 🌟",
    }
    
    # Example user input
    user_input = "I just finished my first marathon! I'm so proud of myself!"
    
    # Detect emotion
    result = detector.detect_emotion(user_input)
    emotion = result['emotion'].lower()
    
    # Get appropriate response
    response = emotion_responses.get(
        emotion,
        f"I see you're feeling {emotion}. Tell me more about it."
    )
    
    print(f"User: {user_input}")
    print(f"Detected Emotion: {emotion}")
    print(f"Chatbot: {response}")
    
    return response


if __name__ == '__main__':
    # Run example
    main()
    
    # Uncomment to see chatbot integration example
    # print("\n\nChatbot Integration Example:")
    # print("="*70)
    # chatbot_response_example()
