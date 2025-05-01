# backend/app/utils/llm.py
import os
import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Import LLM clients
from groq import Groq
import openai
import google.generativeai as genai

# Configure API clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    """Configuration for LLM requests"""
    model: str
    provider: LLMProvider
    temperature: float = 0.7
    max_tokens: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class ProviderError(LLMError):
    """Exception raised when a provider API fails"""
    def __init__(self, provider: str, original_error: Exception):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"{provider.upper()} Error: {str(original_error)}")


class LLMHandler:
    """Handler for LLM requests with failover and retries"""
    
    def __init__(self, default_provider: LLMProvider = LLMProvider.GROQ):
        self.default_provider = default_provider
        self.model_mappings = {
            LLMProvider.GROQ: "llama3-70b-8192-versatile",
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.GEMINI: "gemini-pro"
        }
        
    async def generate(self, 
                      prompt: str, 
                      config: Optional[LLMConfig] = None,
                      failover: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from a prompt with the specified provider
        
        Args:
            prompt: The input prompt
            config: LLM configuration (or use defaults)
            failover: Whether to try alternative providers on failure
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        if not config:
            config = LLMConfig(
                model=self.get_default_model(self.default_provider),
                provider=self.default_provider
            )
        
        metadata = {
            "provider": config.provider,
            "model": config.model,
            "temperature": config.temperature,
            "attempts": 0,
            "start_time": time.time()
        }
        
        # Try primary provider with retries
        for attempt in range(config.retry_attempts):
            metadata["attempts"] += 1
            try:
                result = await self._call_provider(
                    prompt, 
                    config.provider, 
                    config.model, 
                    config.temperature, 
                    config.max_tokens
                )
                metadata["success"] = True
                metadata["duration"] = time.time() - metadata["start_time"]
                return result, metadata
            except Exception as e:
                logger.warning(f"Provider {config.provider} attempt {attempt+1} failed: {str(e)}")
                if attempt < config.retry_attempts - 1:
                    await asyncio.sleep(config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # If primary provider failed and failover is enabled, try alternatives
        if failover:
            backup_providers = [p for p in LLMProvider if p != config.provider]
            for provider in backup_providers:
                try:
                    logger.info(f"Trying failover provider: {provider}")
                    model = self.get_default_model(provider)
                    result = await self._call_provider(
                        prompt, provider, model, config.temperature, config.max_tokens
                    )
                    metadata["failover_provider"] = provider
                    metadata["failover_model"] = model
                    metadata["success"] = True
                    metadata["duration"] = time.time() - metadata["start_time"]
                    return result, metadata
                except Exception as e:
                    logger.warning(f"Failover provider {provider} failed: {str(e)}")
        
        # If we get here, all attempts failed
        metadata["success"] = False
        metadata["duration"] = time.time() - metadata["start_time"]
        error_msg = f"All LLM providers failed to generate a response after {metadata['attempts']} attempts"
        logger.error(error_msg)
        return error_msg, metadata

    async def _call_provider(self, 
                           prompt: str, 
                           provider: LLMProvider, 
                           model: str,
                           temperature: float, 
                           max_tokens: int) -> str:
        """Call the appropriate provider method based on the provider type"""
        try:
            if provider == LLMProvider.GROQ:
                return await self._query_groq(prompt, model, temperature, max_tokens)
            elif provider == LLMProvider.OPENAI:
                return await self._query_openai(prompt, model, temperature, max_tokens)
            elif provider == LLMProvider.GEMINI:
                return await self._query_gemini(prompt, model, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            raise ProviderError(provider.value, e)

    async def _query_groq(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        try:
            completion = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    async def _query_openai(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        try:
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _query_gemini(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                prompt
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def get_default_model(self, provider: LLMProvider) -> str:
        """Get the default model for a provider"""
        return self.model_mappings.get(provider, "gpt-4")