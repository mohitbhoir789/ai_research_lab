"""
Guardrails Module
Implements safety checks and domain restrictions for the AI Research Assistant.
Uses Gemini for more sophisticated topic detection.
"""

import re
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class GuardrailsChecker:
    """Enforces safety checks and domain restrictions using LLM for topic detection."""

    def __init__(self):
        # Keep the allowed domains for reference and documentation
        self.allowed_domains = {
            "computer_science": [
                "algorithms", "data structures", "programming", "software engineering",
                "databases", "operating systems", "networks", "security", "cloud computing",
                "distributed systems", "quantum computing", "artificial intelligence",
                "machine learning", "deep learning", "computer vision", "nlp",
                "robotics", "web development", "systems", "architecture",
                "gradient descent", "optimization", "neural network", "backpropagation",
                "reinforcement learning", "transformer", "attention", "stochastic",
                "generative model", "sentiment analysis", "natural language processing",
                "text mining", "information retrieval", "recommendation systems"
            ],
            "data_science": [
                "statistics", "data analysis", "data mining", "data visualization",
                "big data", "data engineering", "data modeling", "machine learning",
                "deep learning", "neural networks", "predictive analytics",
                "business intelligence", "data warehousing", "etl", "data pipelines",
                "experimentation", "ab testing", "clustering", "regression", "classification",
                "gradient descent", "optimization", "loss function", "hyperparameter",
                "algorithm", "feature engineering", "dimensionality", "model training",
                "backpropagation", "weights", "sgd", "adam", "rmsprop", "momentum",
                "sentiment analysis", "text analytics", "natural language processing",
                "opinion mining", "emotion detection", "data extraction", "information extraction"
            ]
        }
        
        # Initialize Gemini client
        self.gemini_client = None
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai
            logger.info("Gemini client initialized for topic detection")
        else:
            logger.warning("GEMINI_API_KEY not found. Falling back to simple keyword matching for topic detection.")
            
        # Keep OpenAI client as fallback
        self.openai_client = None
        self.async_openai_client = None
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.async_openai_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized as fallback")

    async def check_input(self, text: str) -> Dict[str, Any]:
        """
        Validate user input against safety rules and domain restrictions.
        Uses LLM to detect if the topic is relevant to CS or DS.
        
        Args:
            text: User input text
            
        Returns:
            Dict with validation results
        """
        # Check for appropriate length
        if len(text) > 2000:
            return {
                "passed": False,
                "reason": "too_long",
                "message": "Please keep your queries under 2000 characters."
            }
        
        # Check for code safety (no execution commands)
        text_lower = text.lower()
        dangerous_patterns = [
            r"system\s*\(", r"exec\s*\(", r"eval\s*\(",
            r"os\.", r"subprocess\.", r"bash", r"shell"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower):
                return {
                    "passed": False,
                    "reason": "unsafe_code",
                    "message": "I cannot process potentially unsafe code execution commands."
                }
                
        # Check for domain relevance using LLM
        is_relevant, method_used = await self._check_topic_relevance(text)
        
        if not is_relevant:
            logger.info(f"Topic rejected as off-topic. Method used: {method_used}. Text: '{text[:100]}...'")
            return {
                "passed": False,
                "reason": "off_topic",
                "message": "I can only assist with Computer Science and Data Science topics. Please rephrase your question to focus on these domains."
            }

        logger.info(f"Topic accepted as relevant. Method used: {method_used}")
        return {"passed": True}
        
    async def _check_topic_relevance(self, text: str) -> tuple[bool, str]:
        """
        Check if the text is relevant to Computer Science or Data Science domains.
        Uses Gemini if available, falls back to keyword matching or OpenAI.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (is_relevant, method_used)
        """
        # Always consider common educational/research phrases as relevant
        educational_phrases = ["what is", "how does", "explain", "define", "tell me about", 
                               "describe", "research on", "study of", "concept of", "application of",
                               "example of", "tutorial", "guide", "learn", "understand"]
        
        text_lower = text.lower()
        
        # Check if this is a basic educational question
        for phrase in educational_phrases:
            if phrase in text_lower:
                # For educational questions, do a more lenient check
                # Attempting to detect if this might be related to CS/DS
                potential_cs_ds_terms = [
                    "algorithm", "data", "program", "code", "software", "hardware", 
                    "computer", "network", "system", "machine", "learning", "ai", 
                    "artificial intelligence", "analysis", "model", "predict", 
                    "statistic", "neural", "deep", "mining", "processing",
                    "sentiment", "language", "classification", "recognition",
                    "clustering", "regression", "inference", "computation", "database",
                    "cloud", "security", "encryption", "web", "internet", "api",
                    "framework", "visualization", "dashboard"
                ]
                
                for term in potential_cs_ds_terms:
                    if term in text_lower:
                        # If it looks like an educational question about a CS/DS term,
                        # allow it through without requiring LLM check
                        logger.info(f"Educational question with CS/DS term detected: '{term}'")
                        return True, "educational_pattern"
        
        # Try with Gemini first
        if self.gemini_client:
            try:
                # Use Gemini to classify the topic with improved prompt
                prompt = """You are an AI that determines if a query is related to Computer Science, Data Science, Statistics, Machine Learning, or Data Mining.

The user query is: "{text}"

Respond with YES if the query is related to:
- Computer Science (programming, algorithms, databases, networks, security, etc.)
- Data Science (data analysis, modeling, visualization, etc.)
- Statistics (probability, statistical tests, etc.)
- Machine Learning (neural networks, deep learning, classification, etc.)
- Data Mining (pattern recognition, association rule learning, etc.)

Otherwise respond with NO.
"""
                
                # Configure safety settings
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                # Call Gemini API
                model = self.gemini_client.GenerativeModel(
                    model_name='gemini-1.5-flash',
                    safety_settings=safety_settings
                )
                
                response = model.generate_content(prompt.format(text=text))
                response_text = response.text.strip().lower()
                
                # Check if response indicates the topic is relevant
                is_relevant = "yes" in response_text
                logger.info(f"Gemini topic detection result: {response_text}")
                
                return is_relevant, "gemini"
                
            except Exception as e:
                logger.warning(f"Gemini topic detection failed: {str(e)}")
                # Fall through to next method
        
        # Try with OpenAI as fallback
        if self.async_openai_client:
            try:
                # Create a prompt for OpenAI
                prompt = f"""Determine if this query is related to Computer Science, Data Science, Statistics, Machine Learning, or Data Mining:
                
                "{text}"
                
                Respond with YES or NO."""
                
                response = await self.async_openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI that classifies if queries are related to Computer Science, Data Science, Statistics, Machine Learning, or Data Mining."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )
                
                # Extract answer
                response_text = response.choices[0].message.content.strip().lower()
                is_relevant = "yes" in response_text
                
                logger.info(f"OpenAI topic detection result: {response_text}")
                return is_relevant, "openai"
                
            except Exception as e:
                logger.warning(f"OpenAI topic detection failed: {str(e)}")
                # Fall through to keyword matching

        # Fall back to simple keyword matching
        # Check for keywords related to CS/DS/Stats/ML/Mining
        cs_ds_keywords = set([
            "algorithm", "programming", "code", "software", "computer", 
            "database", "network", "security", "hacking", "cybersecurity",
            "encryption", "web", "app", "developer", "frontend", "backend",
            "javascript", "python", "java", "c++", "html", "css", "api",
            "git", "docker", "cloud", "server", "virtualization", "linux",
            "windows", "unix", "os", "operating system", "debug", "compile",
            "data", "analysis", "analytics", "visualization", "chart", "graph",
            "machine learning", "neural network", "ai", "deep learning",
            "model", "train", "test", "validation", "kaggle", "jupyter",
            "pandas", "numpy", "tensorflow", "pytorch", "scikit", "regression",
            "classification", "clustering", "nlp", "natural language",
            "computer vision", "image processing", "big data", "hadoop",
            "spark", "etl", "dashboard", "bi", "business intelligence",
            "predictive", "prescriptive", "statistics", "probability",
            "hypothesis", "bayes", "statistical", "inference", "data mining",
            "pattern recognition", "association rules", "anomaly detection"
        ])

        # Check for any keyword in the text
        for keyword in cs_ds_keywords:
            if keyword in text_lower:
                logger.info(f"Domain-related keyword found: {keyword}")
                return True, "keyword_match"

        # No relevant keyword found
        return False, "keyword_match"

    async def validate_user_input(self, text: str, custom_domains: List[str] = None) -> bool:
        """
        Validate if user input meets content policies.
        
        Args:
            text: User input text
            custom_domains: Optional list of domain names to override defaults
            
        Returns:
            Boolean indicating if input is valid
        """
        # First check length and safety
        result = await self.check_input(text)
        return result["passed"]
    
    def sanitize_output(self, text: str) -> str:
        """
        Sanitize output before returning to the user.
        Ensures responses don't contain harmful content.
        
        Args:
            text: Response to sanitize
            
        Returns:
            Sanitized response
        """
        # List of patterns to remove
        prohibited_patterns = [
            r"I can't (assist|help) with",
            r"I'm unable to (assist|help) with",
            r"I cannot (assist|help|provide) (with|related to)",
            r"I'm not (able|allowed) to",
            r"I apologize, but I can't",
            r"I'm sorry, but I can't"
        ]
        
        # Check for and remove prohibited patterns
        for pattern in prohibited_patterns:
            text = re.sub(pattern, "I'll help with", text, flags=re.IGNORECASE)
        
        return text