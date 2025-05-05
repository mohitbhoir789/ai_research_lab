"""
Guardrails Module
Implements safety checks and domain restrictions for the AI Research Assistant.
Uses OpenAI for more sophisticated topic detection.
"""

import re
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

class GuardrailsChecker:
    """Enforces safety checks and domain restrictions using OpenAI for topic detection."""

    def __init__(self):
        # Keep the allowed domains for reference and backward compatibility
        self.allowed_domains = {
            "computer_science": [
                "algorithms", "data structures", "programming", "software engineering",
                "databases", "operating systems", "networks", "security", "cloud computing",
                "distributed systems", "quantum computing", "artificial intelligence",
                "machine learning", "deep learning", "computer vision", "nlp",
                "robotics", "web development", "systems", "architecture",
                "gradient descent", "optimization", "neural network", "backpropagation",
                "reinforcement learning", "transformer", "attention", "stochastic",
                "generative model"
            ],
            "data_science": [
                "statistics", "data analysis", "data mining", "data visualization",
                "big data", "data engineering", "data modeling", "machine learning",
                "deep learning", "neural networks", "predictive analytics",
                "business intelligence", "data warehousing", "etl", "data pipelines",
                "experimentation", "ab testing", "clustering", "regression", "classification",
                "gradient descent", "optimization", "loss function", "hyperparameter",
                "algorithm", "feature engineering", "dimensionality", "model training",
                "backpropagation", "weights", "sgd", "adam", "rmsprop", "momentum"
            ]
        }
        
        # Initialize OpenAI client
        self.client = None
        self.async_client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not found. Falling back to simple keyword matching for topic detection.")

    async def check_input(self, text: str) -> Dict[str, Any]:
        """
        Validate user input against safety rules and domain restrictions.
        Uses OpenAI's AI to detect if the topic is relevant to CS or DS.
        Falls back to keyword matching if OpenAI is not available.
        
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
                
        # Check for domain relevance using OpenAI
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
        Uses OpenAI if available, falls back to keyword matching otherwise.
        
        Args:
            text: The text to check
            
        Returns:
            Tuple of (is_relevant, method_used)
        """
        if self.async_client:
            try:
                # Use OpenAI to classify the topic
                response = await self.async_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI that determines if a query is related to Computer Science or Data Science. Respond with 'yes' if it is, or 'no' if it is not. Only respond with 'yes' or 'no'."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0,
                    max_tokens=1
                )
                
                answer = response.choices[0].message.content.strip().lower()
                return answer == "yes", "openai"
                
            except Exception as e:
                logger.error(f"Error using OpenAI for topic detection: {str(e)}")
                # Fall back to keyword matching
                is_relevant = self._keyword_match_topic(text)
                return is_relevant, "keyword_fallback"
        else:
            # Fall back to keyword matching
            is_relevant = self._keyword_match_topic(text)
            return is_relevant, "keyword_only"
            
    def _keyword_match_topic(self, text: str) -> bool:
        """
        Check if the text contains keywords related to Computer Science or Data Science.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text contains relevant keywords, False otherwise
        """
        text_lower = text.lower()
        
        # First check for exact matches on key technical terms
        cs_matches = [topic for topic in self.allowed_domains["computer_science"] if topic in text_lower]
        ds_matches = [topic for topic in self.allowed_domains["data_science"] if topic in text_lower]
        
        # Add more flexible word-boundary matching for multi-word terms
        for topic in self.allowed_domains["computer_science"]:
            if " " in topic and topic not in cs_matches:
                # For multi-word topics, check if all words are present
                words = topic.split()
                if all(word in text_lower for word in words):
                    cs_matches.append(topic)
        
        for topic in self.allowed_domains["data_science"]:
            if " " in topic and topic not in ds_matches:
                # For multi-word topics, check if all words are present
                words = topic.split()
                if all(word in text_lower for word in words):
                    ds_matches.append(topic)

        is_relevant = len(cs_matches) > 0 or len(ds_matches) > 0
        
        if is_relevant:
            if cs_matches:
                logger.info(f"CS keywords matched: {cs_matches}")
            if ds_matches:
                logger.info(f"DS keywords matched: {ds_matches}")
        else:
            logger.info(f"No CS/DS keywords matched in: '{text_lower}'")
            
        return is_relevant

    def sanitize_output(self, text: str) -> str:
        """
        Sanitize agent outputs for safety and relevance.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove any potentially unsafe commands or URLs
        text = re.sub(r"(system|exec|eval)\s*\([^)]*\)", "[REMOVED]", text)
        
        # Ensure proper markdown formatting
        text = self._fix_markdown(text)
        
        return text

    def _fix_markdown(self, text: str) -> str:
        """
        Fix common markdown formatting issues.
        
        Args:
            text: Text to fix
            
        Returns:
            Fixed text
        """
        # Ensure code blocks are properly formatted
        text = re.sub(r'```(\w+)?\s*\n', r'```\1\n', text)
        text = re.sub(r'\n\s*```', r'\n```', text)
        
        # Ensure headers have space after #
        text = re.sub(r'(#+)([^\s])', r'\1 \2', text)
        
        return text