import re
from typing import Tuple, Dict

class IntentDetector:
    """
    Detects user intent to distinguish between greetings, problems, feelings, and questions.
    This helps the chatbot respond appropriately to different message types.
    """
    
    def __init__(self):
        """Initialize the intent detector with pattern definitions."""
        
        # Greeting patterns
        self.greeting_patterns = [
            r'\b(hello|hi|hey|hiya|howdy|greetings|sup|yo)\b',
            r'\b(good\s+(morning|afternoon|evening|day))\b',
            r'\b(what\'?s\s+up|how\'?s\s+it\s+going)\b',
            r'^(hii+|heyy+|helloo+)$',
        ]
        
        # Problem/issue keywords
        self.problem_keywords = [
            'problem', 'issue', 'trouble', 'struggling', 'difficult', 'hard time',
            'challenge', 'concern', 'worry', 'worried', 'anxious', 'stress',
            'can\'t', 'cannot', 'unable', 'don\'t know', 'confused about',
            'help me', 'need advice', 'what should i', 'how do i',
            'relationship', 'work', 'job', 'family', 'friend', 'boss',
            'argument', 'fight', 'conflict', 'mistake', 'failed', 'failing'
        ]
        
        # Feeling/emotion keywords (strong emotional expressions)
        self.feeling_keywords = [
            'feel', 'feeling', 'felt', 'emotion', 'emotional',
            'sad', 'depressed', 'down', 'low', 'blue', 'unhappy',
            'happy', 'excited', 'joyful', 'great', 'amazing', 'wonderful',
            'angry', 'mad', 'furious', 'frustrated', 'annoyed', 'irritated',
            'lonely', 'alone', 'isolated', 'abandoned',
            'overwhelmed', 'exhausted', 'tired', 'drained',
            'scared', 'afraid', 'fearful', 'terrified', 'anxious',
            'hopeless', 'helpless', 'worthless', 'useless'
        ]
        
        # Question patterns
        self.question_patterns = [
            r'\?$',  # Ends with question mark
            r'\b(what|when|where|why|who|how|can|could|would|should|is|are|do|does)\b.*\?',
            r'\b(what|how)\s+(do|can|should|would)\s+i\b',
        ]
        
        # Casual check-in patterns
        self.checkin_patterns = [
            r'\b(just\s+checking\s+in|checking\s+in)\b',
            r'\b(how\s+are\s+you|how\'?re\s+you)\b',
            r'\b(what\'?s\s+new|anything\s+new)\b',
        ]
    
    def detect_intent(self, message: str) -> Tuple[str, float]:
        """
        Detect the primary intent of a user message.
        
        Args:
            message: User's message
            
        Returns:
            Tuple of (intent_type, confidence)
            Intent types: 'greeting', 'problem', 'feeling', 'question', 'checkin', 'neutral'
        """
        if not message or len(message.strip()) == 0:
            return 'neutral', 0.5
        
        message_lower = message.lower().strip()
        message_words = message_lower.split()
        
        # Check for greetings (high priority for short messages)
        if len(message_words) <= 5:
            for pattern in self.greeting_patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    # Pure greeting if it's very short
                    if len(message_words) <= 3:
                        return 'greeting', 0.95
                    else:
                        return 'greeting', 0.85
        
        # Check for check-ins
        for pattern in self.checkin_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return 'checkin', 0.85
        
        # Check for questions
        for pattern in self.question_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                # Could be a problem question or feeling question
                if self._contains_keywords(message_lower, self.problem_keywords):
                    return 'problem', 0.80
                elif self._contains_keywords(message_lower, self.feeling_keywords):
                    return 'feeling', 0.80
                else:
                    return 'question', 0.75
        
        # Check for problems (medium priority)
        problem_score = self._calculate_keyword_score(message_lower, self.problem_keywords)
        
        # Check for feelings (medium priority)
        feeling_score = self._calculate_keyword_score(message_lower, self.feeling_keywords)
        
        # Determine intent based on scores
        # Lower threshold for problem detection to catch more cases
        if problem_score > 0.15 or feeling_score > 0.15:
            if problem_score > feeling_score:
                return 'problem', min(0.9, 0.6 + problem_score)
            else:
                return 'feeling', min(0.9, 0.6 + feeling_score)
        
        # Check if message is very short and vague
        if len(message_words) <= 3 and problem_score == 0 and feeling_score == 0:
            return 'neutral', 0.6
        
        # Default to neutral with moderate confidence
        return 'neutral', 0.5
    
    def _contains_keywords(self, text: str, keywords: list) -> bool:
        """Check if text contains any of the keywords."""
        for keyword in keywords:
            if keyword in text:
                return True
        return False
    
    def _calculate_keyword_score(self, text: str, keywords: list) -> float:
        """
        Calculate a score based on keyword presence.
        
        Args:
            text: Text to analyze
            keywords: List of keywords to search for
            
        Returns:
            Score between 0 and 1
        """
        matches = 0
        text_words = text.split()
        
        for keyword in keywords:
            if keyword in text:
                matches += 1
        
        # Normalize by text length and keyword count
        if len(text_words) == 0:
            return 0.0
        
        # Score based on keyword density
        score = min(1.0, matches / max(len(text_words) * 0.3, 1))
        return score
    
    def get_intent_details(self, message: str) -> Dict:
        """
        Get detailed intent analysis.
        
        Args:
            message: User's message
            
        Returns:
            Dictionary with intent, confidence, and analysis details
        """
        intent, confidence = self.detect_intent(message)
        
        message_lower = message.lower().strip()
        
        return {
            'intent': intent,
            'confidence': confidence,
            'has_greeting': any(re.search(p, message_lower) for p in self.greeting_patterns),
            'has_problem_keywords': self._contains_keywords(message_lower, self.problem_keywords),
            'has_feeling_keywords': self._contains_keywords(message_lower, self.feeling_keywords),
            'is_question': any(re.search(p, message) for p in self.question_patterns),
            'message_length': len(message.split())
        }
