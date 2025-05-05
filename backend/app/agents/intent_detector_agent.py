"""
Intent Detector Agent Module
Specialized agent for detecting user intents and applying guardrails.
"""

import logging
import json
from typing import Dict, Any, Optional, Tuple
from backend.app.agents.agent_core import LLMAgent
from backend.app.utils.llm import LLMConfig, LLMProvider
from backend.app.utils.guardrails import GuardrailsChecker

logger = logging.getLogger(__name__)

class IntentDetectorAgent(LLMAgent):
    """Agent specialized in detecting user intents and applying guardrails."""
    
    def __init__(
        self,
        model: str = "llama3-8b-8192",
        provider: str = "groq",
        agent_id: Optional[str] = None,
        temperature: float = 0.3,  # Lower temperature for more consistent outputs
        max_tokens: int = 150
    ):
        """Initialize the intent detector agent."""
        
        system_prompt = """You are an expert Intent Detection AI that determines:
1. The type of query (chat/research)
2. The depth of response needed (brief/detailed)
3. Whether it's within allowed domains (Computer Science/Data Science)

For chat mode:
- Detect if user wants a brief summary or detailed explanation
- Look for keywords like "explain", "summarize", "tell me about", etc.
- Identify if query needs research paper context

For research mode:
- Detect if user wants to explore existing research or propose new research
- Look for keywords about papers, studies, or new research directions
- Identify if query needs access to research papers or books

Always respond with a structured output:
{
    "mode": "chat" or "research",
    "depth": "brief" or "detailed",
    "requires_papers": true/false,
    "domain_valid": true/false,
    "reason": "Brief explanation of classification"
}"""

        super().__init__(
            model=model,
            provider=provider,
            agent_id=agent_id or "intent_detector",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.guardrails = GuardrailsChecker()
        
        # Add fallback models appropriate for intent detection
        # These models are smaller/faster and good for classification tasks
        self.fallback_models = {
            "groq": ["gemma-7b-it", "llama3-8b-8192"],
            "gemini": ["gemini-1.5-flash"],
            "openai": ["gpt-3.5-turbo"]
        }

    async def detect_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Detect the user's intent and validate against guardrails.
        
        Args:
            user_input: The user's query
            
        Returns:
            Dictionary with intent classification
        """
        # First check guardrails
        check = await self.guardrails.check_input(user_input)
        if not check["passed"]:
            return {
                "mode": "error",
                "depth": "none",
                "requires_papers": False,
                "domain_valid": False,
                "reason": check["reason"]
            }

        prompt = f"""Analyze this user query and classify its intent:

"{user_input}"

Remember:
- Only Computer Science and Data Science topics are valid
- Detect if user wants brief summary or detailed explanation
- Check if research papers/context are needed

Respond with JSON only in the following format:
{{
    "mode": "chat" or "research",
    "depth": "brief" or "detailed",
    "requires_papers": true/false,
    "domain_valid": true/false,
    "reason": "Brief explanation of classification"
}}"""

        # Try each model in sequence until one succeeds
        errors = []
        
        # First try with the configured model
        try:
            config = LLMConfig(
                model=self.model,
                provider=LLMProvider(self.provider),
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Use messages format for better prompt processing
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response, _ = await self.llm.generate(
                config=config,
                extra_params={"messages": messages}
            )
            
            # Parse the JSON response
            intent_data = self._parse_json_response(response)
            if intent_data:
                return intent_data
                
        except Exception as e:
            errors.append(f"Primary model error: {str(e)}")
            logger.warning(f"Intent detection with primary model failed: {str(e)}")
        
        # Try fallback models for this provider
        if self.provider in self.fallback_models:
            for fallback_model in self.fallback_models[self.provider]:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    config = LLMConfig(
                        model=fallback_model,
                        provider=LLMProvider(self.provider),
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    # Use messages format for better prompt processing
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response, _ = await self.llm.generate(
                        config=config,
                        extra_params={"messages": messages}
                    )
                    
                    # Parse the JSON response
                    intent_data = self._parse_json_response(response)
                    if intent_data:
                        return intent_data
                        
                except Exception as e:
                    errors.append(f"Fallback model {fallback_model} error: {str(e)}")
                    logger.warning(f"Intent detection with fallback model {fallback_model} failed: {str(e)}")
        
        # Try with other providers if same provider fallbacks failed
        other_providers = [p for p in ["groq", "gemini", "openai"] if p != self.provider]
        for provider in other_providers:
            if provider in self.fallback_models and self.fallback_models[provider]:
                try:
                    fallback_model = self.fallback_models[provider][0]
                    logger.info(f"Trying provider fallback: {provider}/{fallback_model}")
                    
                    config = LLMConfig(
                        model=fallback_model,
                        provider=LLMProvider(provider),
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    # Use plain text for providers that might not handle messages well
                    response, _ = await self.llm.generate(prompt=prompt, config=config)
                    
                    # Parse the JSON response
                    intent_data = self._parse_json_response(response)
                    if intent_data:
                        return intent_data
                        
                except Exception as e:
                    errors.append(f"Provider {provider} error: {str(e)}")
                    logger.warning(f"Intent detection with provider {provider} failed: {str(e)}")
        
        # If all attempts fail, log the errors and return a default
        logger.error(f"All intent detection attempts failed: {'; '.join(errors)}")
        return {
            "mode": "chat",  # Default to chat mode on error
            "depth": "brief",
            "requires_papers": False,
            "domain_valid": True,
            "reason": "Error in intent detection, defaulting to chat mode"
        }

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling various formatting issues"""
        try:
            # Clean up response to get valid JSON
            response = response.strip()
            
            # Handle markdown code blocks
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
                
            if response.endswith("```"):
                response = response[:-3]
                
            response = response.strip()
            
            # Try to parse JSON
            intent_data = json.loads(response)
            
            # Validate required fields
            required_fields = ["mode", "depth", "requires_papers", "domain_valid"]
            if all(field in intent_data for field in required_fields):
                return intent_data
            else:
                logger.warning(f"Missing required fields in intent data: {intent_data}")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from response: {response[:100]}... Error: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing intent data: {str(e)}")
            return None

    async def run(self, user_input: str) -> Dict[str, Any]:
        """Process user input and detect intent."""
        return await self.detect_intent(user_input)