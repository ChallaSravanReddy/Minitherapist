import json
import random
import os
from typing import Dict, List

class ResponseGenerator:
    """Generate empathetic, supportive responses based on detected emotions."""
    
    def __init__(self, responses_path: str = None):
        """Initialize the response generator.
        
        Args:
            responses_path: Path to responses.json file
        """
        self.responses_path = responses_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'responses.json'
        )
        self.responses = self._load_responses()
        self.last_used = {}  # Track last used responses to avoid repetition
    
    def _load_responses(self) -> Dict:
        """Load response templates from JSON file."""
        try:
            with open(self.responses_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading responses: {e}")
            return {}
    
    def _get_random_component(self, emotion: str, component: str, avoid_last: bool = True) -> str:
        """Get a random component from the response templates.
        
        Args:
            emotion: The detected emotion
            component: Component type (validations, empathy, advice, spiritual, closers)
            avoid_last: Whether to avoid the last used response
            
        Returns:
            Random response component
        """
        if emotion not in self.responses:
            emotion = 'neutral'
        
        components = self.responses[emotion].get(component, [])
        if not components:
            return ""
        
        # Avoid repetition
        key = f"{emotion}_{component}"
        if avoid_last and key in self.last_used and len(components) > 1:
            available = [c for c in components if c != self.last_used[key]]
            choice = random.choice(available)
        else:
            choice = random.choice(components)
        
        self.last_used[key] = choice
        return choice
    
    def generate_response(
        self,
        emotion: str,
        confidence: float,
        user_message: str = "",
        context: Dict = None
    ) -> str:
        """Generate a complete empathetic response.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score of emotion detection
            user_message: Original user message
            context: Additional context (emotion trends, history, etc.)
            
        Returns:
            Complete response string
        """
        # Build response components
        validation = self._get_random_component(emotion, 'validations')
        empathy = self._get_random_component(emotion, 'empathy')
        advice = self._get_random_component(emotion, 'advice')
        spiritual = self._get_random_component(emotion, 'spiritual')
        closer = self._get_random_component(emotion, 'closers')
        
        # Assemble response
        response_parts = []
        
        # Always include validation and empathy
        if validation:
            response_parts.append(validation)
        if empathy:
            response_parts.append(empathy)
        
        # Add advice (80% of the time)
        if advice and random.random() < 0.8:
            response_parts.append(advice)
        
        # Add spiritual guidance (60% of the time, more for certain emotions)
        spiritual_probability = 0.7 if emotion in ['sad', 'confused', 'overwhelmed', 'lonely'] else 0.5
        if spiritual and random.random() < spiritual_probability:
            response_parts.append(spiritual)
        
        # Always include closer
        if closer:
            response_parts.append(closer)
        
        # Join with line breaks for readability
        response = "\n\n".join(response_parts)
        
        return response
    
    def generate_personalized_response(
        self,
        emotion: str,
        confidence: float,
        user_message: str,
        conversation_history: List[str] = None
    ) -> str:
        """Generate a personalized response considering conversation history.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score
            user_message: Current user message
            conversation_history: List of previous messages
            
        Returns:
            Personalized response
        """
        # Analyze context if history available
        context = {}
        if conversation_history:
            # Check if this is a recurring emotion
            context['is_recurring'] = len(conversation_history) > 2
        
        # Generate base response
        response = self.generate_response(emotion, confidence, user_message, context)
        
        # Add personalization for recurring issues
        if context.get('is_recurring') and emotion in ['sad', 'stressed', 'overwhelmed']:
            personalization = random.choice([
                "I notice you've been carrying this for a while.",
                "I see you're still working through this.",
                "I'm here with you through this journey."
            ])
            response = f"{personalization}\n\n{response}"
        
        return response
