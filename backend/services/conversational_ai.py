import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import json
import os
from typing import List, Dict, Tuple
import random

class ConversationalAI:
    """
    Advanced conversational AI using DistilGPT2 for natural, context-aware responses.
    Replaces template-based system with true natural language generation.
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        """Initialize the conversational AI with DistilGPT2 model.
        
        Args:
            model_name: Hugging Face model name (default: distilgpt2)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_name} model...")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✓ Model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
        
        # Conversation memory
        self.conversation_history = {}
        self.max_history = 10
        
        # Load therapy prompts for better responses
        self.therapy_prompts = self._load_therapy_prompts()
    
    def _load_therapy_prompts(self) -> Dict:
        """Load therapy-style conversation prompts."""
        prompts_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'training',
            'therapy_prompts.json'
        )
        
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default prompts if file doesn't exist
        return {
            "sad": [
                "You're a compassionate friend. Someone says: ",
                "As a supportive listener, respond to: "
            ],
            "stressed": [
                "You're a caring friend helping with stress. They say: ",
                "Respond empathetically to: "
            ],
            "angry": [
                "You're a patient friend. Someone is upset and says: ",
                "Respond with understanding to: "
            ],
            "lonely": [
                "You're a warm friend. Someone feeling alone says: ",
                "Respond with compassion to: "
            ],
            "confused": [
                "You're a helpful friend. Someone confused says: ",
                "Respond supportively to: "
            ],
            "overwhelmed": [
                "You're a calming friend. Someone overwhelmed says: ",
                "Respond gently to: "
            ],
            "happy": [
                "You're a joyful friend. Someone happy says: ",
                "Celebrate with them: "
            ],
            "worried": [
                "You're a reassuring friend. Someone worried says: ",
                "Respond with comfort to: "
            ],
            "neutral": [
                "You're a friendly listener. Someone says: ",
                "Respond naturally to: "
            ]
        }
    
    def add_to_history(self, session_id: str, role: str, message: str):
        """Add message to conversation history.
        
        Args:
            session_id: Unique session identifier
            role: 'user' or 'assistant'
            message: The message content
        """
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            'role': role,
            'content': message
        })
        
        # Keep only last N messages
        if len(self.conversation_history[session_id]) > self.max_history * 2:
            self.conversation_history[session_id] = \
                self.conversation_history[session_id][-self.max_history * 2:]
    
    def get_conversation_context(self, session_id: str) -> str:
        """Build conversation context from history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted conversation context
        """
        if session_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[session_id]
        context_parts = []
        
        for msg in history[-6:]:  # Last 3 exchanges
            if msg['role'] == 'user':
                context_parts.append(f"Person: {msg['content']}")
            else:
                context_parts.append(f"Friend: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def generate_response(
        self,
        user_message: str,
        emotion: str = "neutral",
        session_id: str = "default",
        max_length: int = 40,  # Very short to prevent rambling and garbage
        temperature: float = 0.7,  # Lower temperature for more focused responses
        top_p: float = 0.9
    ) -> str:
        """Generate a natural, contextual response.
        
        Args:
            user_message: User's message
            emotion: Detected emotion
            session_id: Session identifier
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response
        """
        if not self.model or not self.tokenizer:
            return self._fallback_response(emotion)
        
        try:
            # Build prompt with context
            context = self.get_conversation_context(session_id)
            
            # Select appropriate prompt based on emotion
            emotion_prompts = self.therapy_prompts.get(emotion, self.therapy_prompts["neutral"])
            prompt_prefix = random.choice(emotion_prompts)
            
            # Construct full prompt with stronger instruction
            # Using a format that encourages dialogue and discourages internet noise
            if context:
                prompt = f"The following is a supportive conversation between a person and a caring friend.\n\n{context}\nPerson: {user_message}\nFriend: "
            else:
                prompt = f"The following is a supportive conversation between a person and a caring friend.\n\nPerson: {user_message}\nFriend: "
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    repetition_penalty=1.2  # Penalize repetition
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response (remove prompt)
            response = generated_text[len(prompt):].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            # Validate response quality - if empty or invalid, use fallback
            if not response or len(response) < 10:
                response = self._get_personality_response(emotion, user_message)
            
            # Add to history
            self.add_to_history(session_id, 'user', user_message)
            self.add_to_history(session_id, 'assistant', response)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._fallback_response(emotion)
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate the generated response with strict quality checks.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response or empty string if invalid
        """
        if not response:
            return ""
        
        # Stop at newlines (often indicates start of new context or garbage)
        if '\n' in response:
            response = response.split('\n')[0]
        
        # Stop at common role markers
        markers = ["Person:", "Friend:", "User:", "Assistant:", "Me:", "You:", "Human:", "AI:"]
        for marker in markers:
            if marker in response:
                response = response.split(marker)[0]
        
        # Remove URLs, handles, and web artifacts
        # Remove anything that looks like a URL
        response = re.sub(r'http\S+|www\.\S+', '', response)
        response = re.sub(r'@\w+|#\w+', '', response)  # Remove mentions and hashtags
        
        # Remove common web artifacts
        web_artifacts = ['&gt;', '&lt;', '&amp;', '&quot;', '&#', 'href=', 'src=', '</', '/>']
        for artifact in web_artifacts:
            if artifact in response:
                return ""  # Invalid response
        
        # Remove any text in brackets or parentheses that looks like metadata
        response = re.sub(r'\[.*?\]', '', response)
        response = re.sub(r'\(.*?http.*?\)', '', response)
        
        # Split into words and filter
        words = response.split()
        clean_words = []
        
        for word in words:
            # Skip words with suspicious patterns (but be less strict)
            if any([
                len(word) > 25,  # Very suspiciously long word
                word.count('.') > 3,  # Many dots (likely URL)
                word.count('/') > 1,  # Multiple slashes (likely URL)
                word.count('_') > 3,  # Many underscores
                word.lower().startswith(('http', 'www', 'ftp'))
            ]):
                continue
            clean_words.append(word)
        
        response = ' '.join(clean_words).strip()
        
        # Validate minimum quality (be less strict)
        if not response or len(response) < 5:
            return ""
        
        # Must have at least 2 words (reduced from 3)
        if len(response.split()) < 2:
            return ""
        
        # Must have at least one letter
        if not re.search(r'[a-zA-Z]', response):
            return ""
        
        # Take first 1-3 complete sentences (max 200 chars, increased from 150)
        sentences = re.split(r'[.!?]+', response)
        cleaned_sentences = []
        total_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short or empty sentences
            if len(sentence) < 3:
                continue
            
            # Skip sentences with suspicious patterns (less strict)
            if any([
                sentence.count(' ') < 1,  # Less than 2 words (reduced from 2)
                len(sentence) > 150,  # Too long for one sentence
            ]):
                continue
            
            # Add sentence if it fits
            if total_length + len(sentence) < 200 and len(cleaned_sentences) < 3:
                cleaned_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        if not cleaned_sentences:
            return ""
        
        result = '. '.join(cleaned_sentences)
        
        # Ensure proper ending punctuation
        if result and not result[-1] in '.!?':
            result += '.'
        
        # Remove any remaining quotes
        result = result.strip('"\'')
        
        # Final validation (less strict)
        if not result or len(result) < 5:
            return ""
        
        return result
    
    def _get_personality_response(self, emotion: str, user_message: str) -> str:
        """Generate personality-driven response with validation → empathy → advice → follow-up structure.
        
        Args:
            emotion: Detected emotion
            user_message: User's message
            
        Returns:
            Structured empathetic response
        """
        import random
        
        # Personality templates following: validate → empathize → advise → follow-up
        templates = {
            "sad": [
                "I hear you, and I'm here with you. That sounds really difficult. Remember that these feelings are temporary, even when they don't feel that way. What's been weighing on you the most?",
                "I can feel the sadness in your words. It's okay to not be okay sometimes. You're being so brave by sharing this. Would you like to talk more about what's making you feel this way?",
                "That sounds really hard, and I'm proud of you for sharing it. Your feelings are completely valid. Sometimes just talking about it can help. I'm right here with you."
            ],
            "stressed": [
                "That sounds overwhelming. Take a deep breath - you're not alone in this. When things feel like too much, try focusing on just one thing at a time. What's the biggest source of stress right now?",
                "I can sense the pressure you're under. It's completely normal to feel stressed. Remember to be gentle with yourself. What would help you feel a bit lighter right now?",
                "That's a lot to carry. You're doing better than you think, even if it doesn't feel that way. Let's break this down together. What feels most urgent?"
            ],
            "angry": [
                "I understand why you're feeling this way. Your feelings are completely valid. It's okay to be angry - it shows you care. What happened that made you feel this way?",
                "I hear the frustration in your words. You have every right to feel angry about this. Sometimes anger is our way of protecting ourselves. Tell me more about what's bothering you.",
                "That would make anyone upset. Your reaction makes complete sense. It's important to acknowledge these feelings. How can I support you through this?"
            ],
            "lonely": [
                "I'm here with you. You're not alone, even when it feels that way. Loneliness is one of the hardest feelings, but you reached out, and that takes courage. What would help you feel more connected?",
                "I hear you, and I want you to know that you matter. Feeling lonely is so difficult, but you're not invisible to me. I'm right here. Tell me what's been going on.",
                "That feeling of being alone is so painful. But you're not alone in this moment - I'm here listening. You're worthy of connection and love. What's been making you feel this way?"
            ],
            "confused": [
                "It's okay to feel uncertain. Life can be confusing, and you don't have to have all the answers right now. Let's talk through this together. What's got you feeling mixed up?",
                "I totally get why you're confused. Sometimes things don't make sense, and that's okay. You're doing your best to figure it out. What's the main thing you're trying to understand?",
                "Feeling lost is completely normal. You're navigating something difficult, and it's okay to not know what to do. I'm here to help you think through it. What are you confused about?"
            ],
            "overwhelmed": [
                "That's a lot to handle. Take it one step at a time - you don't have to do everything at once. You're stronger than you know. What feels most overwhelming right now?",
                "I can feel how much you're carrying. It's okay to feel overwhelmed - it means you care. Let's break this down into smaller pieces. What's one thing we can focus on?",
                "That sounds like so much. Remember to breathe and be kind to yourself. You're doing the best you can. What would help lighten the load a bit?"
            ],
            "happy": [
                "I'm so glad to hear that! Your happiness is wonderful and you deserve to feel this joy. Celebrate this moment! What made this happen?",
                "That's amazing! I'm genuinely happy for you. You deserve all the good things coming your way. Tell me more about what's making you smile!",
                "This is beautiful! Your positive energy is contagious. Enjoy every moment of this happiness. What's bringing you such joy?"
            ],
            "worried": [
                "I hear your concerns. It's natural to worry, but remember that you're capable of handling this. Let's work through this together. What's worrying you the most?",
                "I understand why you're worried. Those feelings are valid, but try not to let them consume you. You've gotten through difficult things before. What's on your mind?",
                "Worry can be so heavy. But you're not facing this alone. Let's talk about what's concerning you and see if we can find some peace together. What's troubling you?"
            ],
            "neutral": [
                "I'm here to listen. Tell me more about what's on your mind. Whatever you're feeling, it's okay to share it with me.",
                "I'm here for you. What would you like to talk about? There's no judgment here, just support.",
                "I'm listening. Feel free to share whatever's on your heart. I'm here to support you."
            ]
        }
        
        responses = templates.get(emotion, templates["neutral"])
        return random.choice(responses)
    
    def _get_greeting_response(self) -> str:
        """Generate a warm greeting response.
        
        Returns:
            Friendly greeting message
        """
        import random
        
        greetings = [
            "Hello! I'm here for you. How are you feeling today?",
            "Hi there! It's good to hear from you. What's on your mind?",
            "Hey! I'm glad you're here. How can I support you today?",
            "Hello! Welcome. I'm here to listen. How are you doing?",
            "Hi! Thanks for reaching out. What would you like to talk about?",
            "Hey there! I'm here for you. How's your day going?",
            "Hello! I'm listening. What's going on with you today?",
            "Hi! It's nice to connect with you. How are you feeling right now?"
        ]
        
        return random.choice(greetings)
    
    def _fallback_response(self, emotion: str) -> str:
        """Simple fallback responses if model fails.
        
        Args:
            emotion: Detected emotion
            
        Returns:
            Simple fallback response
        """
        return self._get_personality_response(emotion, "")
    
    def clear_history(self, session_id: str):
        """Clear conversation history for a session.
        
        Args:
            session_id: Session to clear
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
