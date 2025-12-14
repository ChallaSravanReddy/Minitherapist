import re
from typing import Dict, Tuple

class SafetyLayer:
    """Crisis detection and safety response system."""
    
    def __init__(self):
        """Initialize the safety layer with crisis keywords and resources."""
        
        # Crisis detection keywords (categorized by severity)
        self.high_severity_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'better off dead', 'end it all', 'no reason to live',
            'goodbye cruel world', 'final goodbye'
        ]
        
        self.medium_severity_keywords = [
            'self harm', 'hurt myself', 'cut myself', 'harm myself',
            'punish myself', 'hate myself', 'worthless', 'burden to everyone'
        ]
        
        self.low_severity_keywords = [
            'give up', 'can\'t go on', 'too much pain', 'unbearable',
            'hopeless', 'no point', 'why bother'
        ]
        
        # Crisis helpline resources
        self.helplines = {
            'US': {
                'name': 'National Suicide Prevention Lifeline',
                'phone': '988',
                'text': 'Text HOME to 741741'
            },
            'International': {
                'name': 'International Association for Suicide Prevention',
                'website': 'https://www.iasp.info/resources/Crisis_Centres/'
            }
        }
    
    def check_crisis(self, text: str) -> Tuple[bool, str]:
        """Check if text contains crisis indicators.
        
        Args:
            text: User message to analyze
            
        Returns:
            Tuple of (is_crisis, severity_level)
        """
        text_lower = text.lower()
        
        # Check for high severity
        if any(keyword in text_lower for keyword in self.high_severity_keywords):
            return True, 'high'
        
        # Check for medium severity
        if any(keyword in text_lower for keyword in self.medium_severity_keywords):
            return True, 'medium'
        
        # Check for low severity (multiple indicators needed)
        low_matches = sum(1 for keyword in self.low_severity_keywords if keyword in text_lower)
        if low_matches >= 2:
            return True, 'low'
        
        return False, 'none'
    
    def generate_crisis_response(self, severity: str) -> str:
        """Generate appropriate crisis response based on severity.
        
        Args:
            severity: Crisis severity level (high, medium, low)
            
        Returns:
            Crisis support response
        """
        if severity == 'high':
            response = """I'm really concerned about what you're sharing, and I want you to know that you matter deeply. What you're feeling right now is temporary, even though it doesn't feel that way.

**Please reach out for immediate support:**

🆘 **National Suicide Prevention Lifeline: 988**
📱 **Crisis Text Line: Text HOME to 741741**
🌍 **International resources: https://www.iasp.info/resources/Crisis_Centres/**

You deserve real support from trained professionals who can help you through this. Please don't face this alone.

I'm here to listen, but I'm not equipped to provide the level of support you need right now. Your life has value, and there are people who want to help you see that. 💙"""
        
        elif severity == 'medium':
            response = """I hear how much pain you're in, and I'm concerned about you. The thoughts you're having about harming yourself are serious, and you deserve real support.

**Please consider reaching out:**

🆘 **Crisis Lifeline: 988**
📱 **Text Support: Text HOME to 741741**

Talking to a trained counselor can make a real difference. They're available 24/7 and want to help.

You don't have to go through this alone. Your feelings are valid, but there are healthier ways to cope with this pain. Please reach out for help. 💙"""
        
        else:  # low severity
            response = """I can hear how difficult things feel right now. When everything seems hopeless, it's important to remember that these feelings, as intense as they are, are temporary.

If you're struggling, please consider talking to someone:

📞 **Crisis Support: 988**
💬 **Text Support: Text HOME to 741741**

You deserve support, and there are people trained to help you through this. You don't have to carry this alone.

I'm here to listen, but please know that professional support can make a real difference. You matter. 💙"""
        
        return response
    
    def get_supportive_followup(self) -> str:
        """Get a gentle follow-up message after crisis response.
        
        Returns:
            Follow-up message
        """
        followups = [
            "I'm still here with you. How are you feeling right now?",
            "Please know that I care about your wellbeing. Have you been able to reach out for support?",
            "I'm here to listen. Would you like to talk about what's going on?",
            "You're not alone in this. I'm here, and there are people who want to help."
        ]
        
        import random
        return random.choice(followups)
    
    def log_crisis_event(self, severity: str, message_preview: str):
        """Log crisis events for monitoring (privacy-conscious).
        
        Args:
            severity: Crisis severity level
            message_preview: First few words of message (for context)
        """
        # In production, this would log to a secure system
        # For now, just print to console
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        print(f"[CRISIS ALERT] {timestamp} - Severity: {severity} - Preview: {message_preview[:30]}...")
