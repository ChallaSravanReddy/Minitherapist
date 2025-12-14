# 🧠 Mini Therapist - Emotional Support & Spiritual Guidance Chatbot

A compassionate AI chatbot that provides emotional support, spiritual guidance, and friendly advice. Built with Python Flask backend (ML-powered emotion detection) and modern vanilla JavaScript frontend.

## ✨ Features

- **🎭 Emotion Detection**: ML-based classification of 9 emotions (sad, stressed, angry, lonely, confused, overwhelmed, happy, worried, neutral)
- **💬 Empathetic Responses**: Context-aware, supportive responses with validation, empathy, advice, and spiritual guidance
- **🆘 Safety Layer**: Crisis detection with appropriate resources and helpline information
- **✨ Daily Affirmations**: 100+ uplifting affirmations for self-empowerment
- **🙏 Spiritual Quotes**: 100+ quotes from diverse traditions (Buddhism, Stoicism, Sufism, etc.)
- **📊 Mood Tracker**: Visual tracking of emotional patterns over time
- **🌓 Dark/Light Mode**: Beautiful themes with glassmorphism effects
- **💾 Local Storage**: Browser-based conversation and mood history

## 🏗️ Architecture

```
Minitherapist/
├── backend/                    # Python Flask API
│   ├── app.py                 # Main Flask application
│   ├── train_model.py         # ML model training script
│   ├── requirements.txt       # Python dependencies
│   ├── models/
│   │   ├── emotion_classifier.py
│   │   └── trained_model.pkl  # Trained scikit-learn model
│   ├── services/
│   │   ├── response_generator.py
│   │   ├── safety_layer.py
│   │   └── nlp_processor.py
│   └── data/
│       ├── training/emotions_training.json
│       ├── responses.json
│       ├── affirmations.json
│       └── quotes.json
└── frontend/                   # Vanilla JavaScript UI
    ├── index.html
    ├── css/styles.css
    └── js/
        ├── app.js
        ├── api.js
        └── storage.js
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Modern web browser

### Installation

1. **Clone or navigate to the project directory**

2. **Install Python dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **Download spaCy language model** (optional, for better NLP)
```bash
python -m spacy download en_core_web_sm
```

4. **Train the emotion classification model**
```bash
python train_model.py
```

This will:
- Load 200 training examples
- Train a Logistic Regression classifier with TF-IDF features
- Save the model to `models/trained_model.pkl`
- Display accuracy metrics

### Running the Application

1. **Start the Flask backend**
```bash
python app.py
```

The backend will start on `http://localhost:5000`

2. **Open the frontend**

Simply open `frontend/index.html` in your web browser, or use a local server:

```bash
cd frontend
python -m http.server 8000
```

Then visit `http://localhost:8000`

## 🎯 Usage

### Chat Interface

1. Type your message in the input box
2. Press Enter or click the send button
3. Receive empathetic, supportive responses
4. Your conversation is saved locally

### Features

- **✨ Daily Affirmation**: Click the sparkle icon for an uplifting affirmation
- **🙏 Spiritual Quote**: Click the prayer icon for wisdom from diverse traditions
- **📊 Mood Tracker**: Click the chart icon to see your emotional patterns
- **🌓 Theme Toggle**: Click the moon/sun icon to switch between dark and light mode

### Crisis Support

If the chatbot detects crisis keywords (suicide, self-harm, etc.), it will:
- Provide supportive, non-judgmental response
- Display crisis helpline resources
- Encourage seeking professional help

**Important**: This is NOT a replacement for professional mental health support.

## 🔧 API Endpoints

- `POST /api/chat` - Send message, receive response
- `GET /api/affirmation` - Get random affirmation
- `GET /api/quote` - Get random spiritual quote
- `POST /api/mood-history` - Get mood tracking data
- `GET /api/health` - Health check

## 🎨 Design Features

- **Glassmorphism**: Modern frosted glass effects
- **Smooth Animations**: Micro-interactions and transitions
- **Responsive Design**: Works on mobile and desktop
- **Premium Aesthetics**: Gradient accents, custom scrollbars
- **Accessibility**: Semantic HTML, proper contrast ratios

## 📊 ML Model Details

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: Unigrams and bigrams, max 1000 features
- **Training Data**: 200 labeled examples across 9 emotions
- **Fallback**: Rule-based keyword matching if model unavailable
- **Accuracy**: ~85-90% on test set (varies with training data)

## 🔮 Future Enhancements

- [ ] Database integration (SQLite/PostgreSQL)
- [ ] User authentication
- [ ] Export conversation history
- [ ] More advanced NLP (transformer models)
- [ ] Voice input/output
- [ ] Mobile app (React Native)
- [ ] Multi-language support

## 🛡️ Privacy & Safety

- **Local Storage**: All data stored in browser (no server-side storage currently)
- **No Tracking**: No analytics or user tracking
- **Crisis Detection**: Automatic detection with resource provision
- **Disclaimer**: Not a replacement for professional mental health services

## 📝 License

This project is for educational and personal use.

## 🙏 Acknowledgments

- Spiritual quotes from various wisdom traditions
- Crisis resources from national helplines
- Built with love and compassion 💙

---

**Remember**: You are worthy of love, support, and happiness. If you're struggling, please reach out to a mental health professional or crisis helpline. You're not alone. 💙
