from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import os
import torch 

# Import our custom modules
from models.emotion_classifier import EmotionClassifier
from services.response_generator import ResponseGenerator
from services.safety_layer import SafetyLayer
from services.conversational_ai import ConversationalAI
from services.sentiment_analyzer import SentimentAnalyzer
from services.intent_detector import IntentDetector


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize services
print("\n" + "="*50)
print("🧠 Mini Therapist Backend Starting...")
print("="*50)

emotion_classifier = EmotionClassifier()
response_generator = ResponseGenerator()  # Keep as fallback
safety_layer = SafetyLayer()
intent_detector = IntentDetector()

# Initialize Conversational AI (this may take a moment)
print("\n🤖 Loading Conversational AI...")
conversational_ai = ConversationalAI()
sentiment_analyzer = SentimentAnalyzer()
print("="*50 + "\n")

# Load affirmations and quotes
def load_json_file(filename):
    """Load JSON file from data directory."""
    file_path = os.path.join(os.path.dirname(__file__), 'data', filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

affirmations = load_json_file('affirmations.json')
quotes = load_json_file('quotes.json')

# In-memory conversation storage (for demo - replace with database later)
conversations = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint - process user message and return bot response."""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversation_history = conversations[session_id]
        
        # Check for crisis first
        is_crisis, severity = safety_layer.check_crisis(user_message)
        
        if is_crisis:
            # Log crisis event
            safety_layer.log_crisis_event(severity, user_message)
            
            # Generate crisis response
            bot_response = safety_layer.generate_crisis_response(severity)
            
            # Store in conversation
            conversations[session_id].append({
                'user': user_message,
                'bot': bot_response,
                'emotion': 'crisis',
                'severity': severity
            })
            
            return jsonify({
                'response': bot_response,
                'emotion': 'crisis',
                'severity': severity,
                'is_crisis': True
            })
        
        # Detect intent first (greeting, problem, feeling, etc.)
        intent, intent_confidence = intent_detector.detect_intent(user_message)
        
        # Detect emotion
        emotion, confidence = emotion_classifier.predict(user_message)
        
        # Generate response based on intent
        try:
            # Handle greetings specially
            if intent == 'greeting':
                bot_response = conversational_ai._get_greeting_response()
            # Handle check-ins
            elif intent == 'checkin':
                bot_response = "I'm doing well, thank you for asking! More importantly, how are you doing? I'm here to listen and support you."
            # Handle problems and feelings with emotion-aware responses
            else:
                bot_response = conversational_ai.generate_response(
                    user_message=user_message,
                    emotion=emotion,
                    session_id=session_id,
                    max_length=40,  # Short to prevent garbage
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Validate response - ensure no random characters
            if not bot_response or len(bot_response) < 10 or not any(c.isalpha() for c in bot_response):
                raise ValueError("Invalid response generated")
                
        except Exception as e:
            print(f"AI generation error, using fallback: {e}")
            # Fallback to template-based if AI fails
            if intent == 'greeting':
                bot_response = "Hello! I'm here for you. How are you feeling today?"
            else:
                bot_response = response_generator.generate_personalized_response(
                    emotion=emotion,
                    confidence=confidence,
                    user_message=user_message,
                    conversation_history=[msg['user'] for msg in conversation_history]
                )
        
        # Store in conversation
        conversations[session_id].append({
            'user': user_message,
            'bot': bot_response,
            'emotion': emotion,
            'confidence': confidence
        })
        
        # Keep only last 10 messages
        if len(conversations[session_id]) > 10:
            conversations[session_id] = conversations[session_id][-10:]
        
        # Get sentiment analysis
        sentiment = sentiment_analyzer.analyze(user_message)
        
        return jsonify({
            'response': bot_response,
            'emotion': emotion,
            'confidence': confidence,
            'intent': intent,
            'intent_confidence': intent_confidence,
            'is_crisis': False,
            'sentiment': sentiment['sentiment'],
            'ai_generated': True
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/affirmation', methods=['GET'])
def get_affirmation():
    """Get a random daily affirmation."""
    try:
        if not affirmations:
            return jsonify({'error': 'No affirmations available'}), 500
        
        affirmation = random.choice(affirmations)
        return jsonify({'affirmation': affirmation})
        
    except Exception as e:
        print(f"Error in affirmation endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/quote', methods=['GET'])
def get_quote():
    """Get a random spiritual quote."""
    try:
        if not quotes:
            return jsonify({'error': 'No quotes available'}), 500
        
        quote = random.choice(quotes)
        return jsonify(quote)
        
    except Exception as e:
        print(f"Error in quote endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/mood-history', methods=['POST'])
def get_mood_history():
    """Get mood tracking history for a session."""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in conversations:
            return jsonify({'history': []})
        
        # Extract emotion history
        history = []
        for msg in conversations[session_id]:
            if msg.get('emotion') != 'crisis':
                history.append({
                    'emotion': msg.get('emotion'),
                    'confidence': msg.get('confidence', 0)
                })
        
        return jsonify({'history': history})
        
    except Exception as e:
        print(f"Error in mood-history endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': emotion_classifier.model is not None,
        'intent_detector_loaded': intent_detector is not None,
        'affirmations_count': len(affirmations),
        'quotes_count': len(quotes)
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'message': 'Mini Therapist API',
        'version': '2.0.0',
        'endpoints': {
            'chat': '/api/chat',
            'affirmation': '/api/affirmation',
            'quote': '/api/quote',
            'mood_history': '/api/mood-history',
            'health': '/api/health'
        }
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🧠 Mini Therapist Backend Ready!")
    print("="*50)
    print(f"✓ Affirmations loaded: {len(affirmations)}")
    print(f"✓ Quotes loaded: {len(quotes)}")
    print(f"✓ Emotion Model loaded: {emotion_classifier.model is not None}")
    print(f"✓ Intent Detector loaded: {intent_detector is not None}")
    print(f"✓ Conversational AI loaded: {conversational_ai.model is not None}")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
