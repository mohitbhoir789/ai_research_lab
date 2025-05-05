"""
LLM Handler Module
Unified interface for multiple LLM providers with retry and failover mechanisms.
"""

import os
import time
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List
import logging
from dotenv import load_dotenv

# Import provider SDKs
from groq import AsyncGroq
from openai import AsyncOpenAI
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"

class LLMConfig:
    """Configuration for LLM requests."""
    
    def __init__(
        self,
        model: str,
        provider: LLMProvider,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        system_instruction: Optional[str] = None
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.system_instruction = system_instruction

class LLMHandler:
    """Unified interface for multiple LLM providers with retry and failover."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize LLM clients for each provider."""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize clients
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Provider-specific model mappings
        self.model_mappings = {
            LLMProvider.GROQ: {
                "default": "mixtral-8x7b-32768",
                "fast": "llama-3.1-8b-instant",
                "code": "codellama-34b",
                "guard": "llama-guard-3-8b",
                "large": "llama-3.3-70b-versatile",
                "context": "llama3-70b-8192",
                "small": "gemma2-9b-it"
            },
            LLMProvider.OPENAI: {
                "default": "gpt-4-turbo-preview",
                "fast": "gpt-3.5-turbo",
                "code": "gpt-4-turbo-preview",
                "large": "gpt-4-1106-preview"
            },
            LLMProvider.GEMINI: {
                "default": "gemini-1.5-flash",
                "fast": "gemini-1.5-flash",
                "pro": "gemini-1.5-pro",
                "large": "gemini-1.5-pro"
            }
        }
        
        # Track provider availability
        self.provider_available = {
            LLMProvider.GROQ: True,
            LLMProvider.OPENAI: True,
            LLMProvider.GEMINI: True
        }

    async def generate(
        self,
        prompt: Optional[str] = None,
        config: LLMConfig = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text using the specified LLM provider with retry and failover.
        
        Args:
            prompt: Text prompt (if not using chat format)
            config: LLM configuration
            extra_params: Additional provider-specific parameters (e.g., messages for chat)
            
        Returns:
            Tuple of (generated text, metadata)
        """
        config = config or LLMConfig(
            model=self.model_mappings[LLMProvider.GROQ]["default"],
            provider=LLMProvider.GROQ
        )
        
        extra_params = extra_params or {}
        
        # If the configured provider is known to be unavailable, try failover immediately
        if not self.provider_available[config.provider]:
            logger.warning(f"Provider {config.provider} is known to be unavailable, trying failover immediately")
            return await self._failover_generation(
                prompt, config, extra_params, 
                Exception(f"Provider {config.provider} is marked as unavailable")
            )
            
        # Try primary provider
        for attempt in range(self.max_retries):
            try:
                return await self._generate_with_provider(
                    prompt, config, extra_params
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                
                # Mark the provider as unavailable for future requests
                self.provider_available[config.provider] = False
                logger.warning(f"Marked provider {config.provider} as unavailable")
                
                # Try failover to different provider
                return await self._failover_generation(
                    prompt, config, extra_params, original_error=e
                )

    async def _generate_with_provider(
        self,
        prompt: Optional[str],
        config: LLMConfig,
        extra_params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using a specific provider."""
        
        if config.provider == LLMProvider.GROQ:
            return await self._generate_groq(prompt, config, extra_params)
        elif config.provider == LLMProvider.OPENAI:
            return await self._generate_openai(prompt, config, extra_params)
        elif config.provider == LLMProvider.GEMINI:
            return await self._generate_gemini(prompt, config, extra_params)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    async def _generate_groq(
        self,
        prompt: Optional[str],
        config: LLMConfig,
        extra_params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using Groq."""
        
        try:
            if "messages" in extra_params:
                # Chat completion
                response = await self.groq_client.chat.completions.create(
                    model=config.model,
                    messages=extra_params["messages"],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens or 2000,
                    top_p=config.top_p
                )
                return response.choices[0].message.content, {
                    "provider": "groq",
                    "model": config.model,
                    "finish_reason": response.choices[0].finish_reason
                }
            else:
                # For text completion, we'll use chat interface with a user message
                # as Groq doesn't have a completions endpoint
                messages = [{"role": "user", "content": prompt or ""}]
                response = await self.groq_client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens or 2000,
                    top_p=config.top_p
                )
                return response.choices[0].message.content, {
                    "provider": "groq",
                    "model": config.model,
                    "finish_reason": response.choices[0].finish_reason
                }
        except Exception as e:
            logger.error(f"Groq generation error: {str(e)}")
            raise

    async def _generate_openai(
        self,
        prompt: Optional[str],
        config: LLMConfig,
        extra_params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using OpenAI."""
        
        try:
            if "messages" in extra_params:
                # Chat completion
                response = await self.openai_client.chat.completions.create(
                    model=config.model,
                    messages=extra_params["messages"],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens or 2000,
                    top_p=config.top_p,
                    presence_penalty=config.presence_penalty,
                    frequency_penalty=config.frequency_penalty
                )
                return response.choices[0].message.content, {
                    "provider": "openai",
                    "model": config.model,
                    "finish_reason": response.choices[0].finish_reason
                }
            else:
                # For new OpenAI API, use chat interface as legacy completions are deprecated
                messages = [{"role": "user", "content": prompt or ""}]
                response = await self.openai_client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens or 2000,
                    top_p=config.top_p,
                    presence_penalty=config.presence_penalty,
                    frequency_penalty=config.frequency_penalty
                )
                return response.choices[0].message.content, {
                    "provider": "openai",
                    "model": config.model,
                    "finish_reason": response.choices[0].finish_reason
                }
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            raise

    async def _generate_gemini(
        self,
        prompt: Optional[str],
        config: LLMConfig,
        extra_params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using Google's Gemini."""
        
        try:
            # Prepare system instruction if available
            system_instruction = config.system_instruction or None
            
            # Set up generation parameters
            generation_config = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_output_tokens": config.max_tokens or 2000
            }
            
            # Handle messages vs direct prompt
            if "messages" in extra_params:
                # Format messages for Gemini API
                gemini_messages = []
                for msg in extra_params["messages"]:
                    if msg.get("role") == "user":
                        gemini_messages.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})
                    elif msg.get("role") == "assistant":
                        gemini_messages.append({"role": "model", "parts": [{"text": msg.get("content", "")}]})
                    # System message is handled separately in gemini API
                    elif msg.get("role") == "system" and not system_instruction:
                        system_instruction = msg.get("content", "")
                
                # Initialize Gemini model
                model = genai.GenerativeModel(
                    model_name=config.model,
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
                
                # Create a chat session
                chat = model.start_chat()
                
                # Send the conversation history and get response
                response = await asyncio.to_thread(
                    chat.send_message, 
                    gemini_messages[-1]["parts"][0]["text"] if gemini_messages else "Hello"
                )
            else:
                # Direct text completion
                model = genai.GenerativeModel(
                    model_name=config.model,
                    generation_config=generation_config,
                    system_instruction=system_instruction
                )
                
                # Generate content
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt or ""
                )
            
            # Extract text from response
            if hasattr(response, 'text'):
                result_text = response.text
            else:
                result_text = response.candidates[0].content.parts[0].text
                
            return result_text, {
                "provider": "gemini",
                "model": config.model,
                "finish_reason": "stop"  # Gemini doesn't provide this info directly
            }
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            raise

    async def _failover_generation(
        self,
        prompt: Optional[str],
        config: LLMConfig,
        extra_params: Dict[str, Any],
        original_error: Exception
    ) -> Tuple[str, Dict[str, Any]]:
        """Try generation with failover providers."""
        
        # Define failover order - prioritize by speed and availability
        failover_providers = []
        
        # Add available providers to the failover list
        for provider in [LLMProvider.OPENAI, LLMProvider.GROQ, LLMProvider.GEMINI]:
            if provider != config.provider and self.provider_available[provider]:
                failover_providers.append(provider)
        
        # If no providers are available, try all of them one last time
        if not failover_providers:
            failover_providers = [p for p in [LLMProvider.OPENAI, LLMProvider.GROQ, LLMProvider.GEMINI] 
                                if p != config.provider]
            # Reset availability for this emergency attempt
            for provider in failover_providers:
                self.provider_available[provider] = True
        
        # Try each failover provider
        last_error = original_error
        for provider in failover_providers:
            try:
                logger.info(f"Trying failover with provider: {provider}")
                # Map the model priority (e.g., "default", "fast") to equivalent in failover provider
                model_priority = self._get_model_priority(config.provider, config.model)
                failover_model = self.model_mappings[provider].get(
                    model_priority, 
                    self.model_mappings[provider]["default"]
                )
                
                failover_config = LLMConfig(
                    model=failover_model,
                    provider=provider,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    system_instruction=config.system_instruction
                )
                
                result = await self._generate_with_provider(
                    prompt, failover_config, extra_params
                )
                
                # If successful, mark the provider as available again
                self.provider_available[provider] = True
                
                return result
            except Exception as e:
                logger.warning(f"Failover to {provider} failed: {str(e)}")
                # Mark this provider as unavailable too
                self.provider_available[provider] = False
                last_error = e
                continue
        
        # If all failovers failed, raise the last error
        raise Exception(
            f"All providers failed. Original error: {str(original_error)}. "
            f"Last failover error: {str(last_error)}"
        )
    
    def _get_model_priority(self, provider: LLMProvider, model_name: str) -> str:
        """Determine the model priority (default, fast, code, etc.) based on the model name"""
        provider_models = self.model_mappings[provider]
        for priority, model in provider_models.items():
            if model == model_name:
                return priority
        return "default"  # Default fallback

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a text prompt for providers that don't support chat."""
        formatted_messages = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_messages.append(f"{role}: {content}")
        return "\n".join(formatted_messages)