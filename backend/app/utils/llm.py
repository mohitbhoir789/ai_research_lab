# backend/app/utils/llm.py
import os
import asyncio
import logging
import time
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Import LLM clients
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    logger.warning("Groq package not available")
    GROQ_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available")
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Google Generative AI package not available")
    GEMINI_AVAILABLE = False


# Initialize clients if possible
groq_client = None
if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

openai_client = None
if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"
    
    @classmethod
    def from_string(cls, provider_str: str) -> 'LLMProvider':
        """Convert string to enum value with validation"""
        try:
            return cls(provider_str.lower())
        except ValueError:
            logger.warning(f"Unknown provider '{provider_str}', defaulting to OPENAI")
            return cls.OPENAI


class ModelCapability(Enum):
    """Capabilities that models might have"""
    CODE = auto()           # Strong at code generation
    REASONING = auto()      # Strong at logical reasoning
    CREATIVITY = auto()     # Good for creative content
    MATH = auto()           # Good at mathematical reasoning
    KNOWLEDGE = auto()      # Has extensive knowledge


@dataclass
class ModelInfo:
    """Information about an LLM model"""
    name: str
    provider: LLMProvider
    max_tokens: int
    capabilities: List[ModelCapability] = field(default_factory=list)
    suggested_temperature: float = 0.7
    relative_speed: int = 5  # 1-10 scale, 10 being fastest
    relative_quality: int = 5  # 1-10 scale, 10 being highest quality


@dataclass
class LLMConfig:
    """Configuration for LLM requests"""
    model: str
    provider: LLMProvider
    temperature: float = 0.7
    max_tokens: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    timeout: float = 30.0  # seconds
    stream: bool = False
    system_prompt: Optional[str] = None
    stop_sequences: List[str] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class ProviderError(LLMError):
    """Exception raised when a provider API fails"""
    def __init__(self, provider: LLMProvider, original_error: Exception):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"{provider.value.upper()} Error: {str(original_error)}")


class ModelNotAvailableError(LLMError):
    """Exception raised when a model is not available"""
    def __init__(self, model: str, provider: LLMProvider):
        self.model = model
        self.provider = provider
        super().__init__(f"Model {model} not available for provider {provider.value}")


class LLMHandler:
    """Enhanced handler for LLM requests with failover, retries, and metrics"""
    
    def __init__(self, default_provider: LLMProvider = LLMProvider.OPENAI):
        self.default_provider = default_provider
        
        # Map of provider to available models
        self.model_mappings = {
            LLMProvider.GROQ: [
                ModelInfo("llama3-70b-8192", LLMProvider.GROQ, 8192, 
                         [ModelCapability.REASONING, ModelCapability.CODE], 0.7, 7, 8),
                ModelInfo("llama3-8b-8192", LLMProvider.GROQ, 8192,
                         [ModelCapability.CODE], 0.7, 9, 6),
                ModelInfo("mixtral-8x7b-32768", LLMProvider.GROQ, 32768,
                         [ModelCapability.REASONING, ModelCapability.KNOWLEDGE], 0.7, 6, 7)
            ],
            LLMProvider.OPENAI: [
                ModelInfo("gpt-4", LLMProvider.OPENAI, 8192, 
                         [ModelCapability.REASONING, ModelCapability.CODE, 
                          ModelCapability.KNOWLEDGE, ModelCapability.MATH], 0.7, 5, 9),
                ModelInfo("gpt-3.5-turbo", LLMProvider.OPENAI, 4096,
                         [ModelCapability.CODE, ModelCapability.CREATIVITY], 0.7, 8, 7)
            ],
            LLMProvider.GEMINI: [
                ModelInfo("gemini-2.0-flash", LLMProvider.GEMINI, 8192,
                         [ModelCapability.REASONING, ModelCapability.CODE], 0.7, 6, 8)
            ]
        }
        
        # Check which providers are actually available
        self.available_providers = self._get_available_providers()
        logger.info(f"Available LLM providers: {[p.value for p in self.available_providers]}")
        
        # Metrics tracking
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        
        # Setup callbacks for request events
        self.pre_request_callbacks: List[Callable] = []
        self.post_request_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
    def _get_available_providers(self) -> List[LLMProvider]:
        """Check which providers are actually available based on clients and API keys"""
        available = []
        
        if GROQ_AVAILABLE and groq_client:
            available.append(LLMProvider.GROQ)
            
        if OPENAI_AVAILABLE and openai_client:
            available.append(LLMProvider.OPENAI)
            
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            available.append(LLMProvider.GEMINI)
            
        return available
        
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for request events
        
        Args:
            event_type: 'pre_request', 'post_request', or 'error'
            callback: Function to call on event
        """
        if event_type == 'pre_request':
            self.pre_request_callbacks.append(callback)
        elif event_type == 'post_request':
            self.post_request_callbacks.append(callback)
        elif event_type == 'error':
            self.error_callbacks.append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def get_default_model(self, provider: LLMProvider) -> str:
        """Get the default model for a provider"""
        provider_models = self.model_mappings.get(provider, [])
        if not provider_models:
            raise ValueError(f"No models available for provider {provider.value}")
        return provider_models[0].name

    def get_model_info(self, model_name: str, provider: LLMProvider) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        provider_models = self.model_mappings.get(provider, [])
        for model_info in provider_models:
            if model_info.name == model_name:
                return model_info
        return None
    
    def find_model_by_capability(self, capability: ModelCapability, 
                                prefer_provider: Optional[LLMProvider] = None) -> ModelInfo:
        """
        Find the best model with a specific capability
        
        Args:
            capability: The capability to look for
            prefer_provider: Optional preferred provider
            
        Returns:
            Best matching model info
        """
        candidates = []
        
        # First check preferred provider if specified
        if prefer_provider:
            for model in self.model_mappings.get(prefer_provider, []):
                if capability in model.capabilities:
                    candidates.append(model)
            
            if candidates:
                # Sort by quality and return the best
                candidates.sort(key=lambda m: m.relative_quality, reverse=True)
                return candidates[0]
        
        # If no preferred provider or no matches found, check all providers
        for provider, models in self.model_mappings.items():
            if provider in self.available_providers:
                for model in models:
                    if capability in model.capabilities:
                        candidates.append(model)
        
        if not candidates:
            raise ValueError(f"No models found with capability {capability}")
            
        # Sort by quality and return the best
        candidates.sort(key=lambda m: m.relative_quality, reverse=True)
        return candidates[0]
        
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
            provider = self.default_provider if self.default_provider in self.available_providers else self.available_providers[0]
            config = LLMConfig(
                model=self.get_default_model(provider),
                provider=provider
            )
        
        # Increment request counter
        self.request_count += 1
        
        metadata = {
            "request_id": self.request_count,
            "provider": config.provider.value,
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "attempts": 0,
            "start_time": time.time()
        }
        
        # Validate provider availability
        if config.provider not in self.available_providers:
            logger.warning(f"Requested provider {config.provider.value} not available")
            if failover:
                logger.info(f"Falling back to available provider: {self.available_providers[0].value}")
                config.provider = self.available_providers[0]
                config.model = self.get_default_model(config.provider)
                metadata["provider"] = config.provider.value
                metadata["model"] = config.model
                metadata["fallback_reason"] = "provider_not_available"
            else:
                error_msg = f"Provider {config.provider.value} not available and failover disabled"
                metadata["success"] = False
                metadata["error"] = error_msg
                metadata["duration"] = time.time() - metadata["start_time"]
                return error_msg, metadata
        
        # Trigger pre-request callbacks
        for callback in self.pre_request_callbacks:
            try:
                callback(prompt=prompt, config=config, metadata=metadata)
            except Exception as e:
                logger.error(f"Error in pre-request callback: {str(e)}")
        
        # Try primary provider with retries
        last_error = None
        for attempt in range(config.retry_attempts):
            metadata["attempts"] += 1
            try:
                result = await self._call_provider(
                    prompt, 
                    config.provider, 
                    config.model, 
                    config.temperature, 
                    config.max_tokens,
                    config.system_prompt,
                    config.stop_sequences,
                    config.extra_params,
                    config.timeout,
                    config.stream
                )
                metadata["success"] = True
                metadata["duration"] = time.time() - metadata["start_time"]
                
                # Trigger post-request callbacks
                for callback in self.post_request_callbacks:
                    try:
                        callback(prompt=prompt, result=result, metadata=metadata)
                    except Exception as e:
                        logger.error(f"Error in post-request callback: {str(e)}")
                
                # Update metrics
                self.total_latency += metadata["duration"]
                
                return result, metadata
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {config.provider.value} attempt {attempt+1} failed: {str(e)}")
                
                # Trigger error callbacks
                for callback in self.error_callbacks:
                    try:
                        callback(prompt=prompt, error=e, metadata=metadata)
                    except Exception as e:
                        logger.error(f"Error in error callback: {str(e)}")
                
                if attempt < config.retry_attempts - 1:
                    # Use exponential backoff
                    backoff_time = config.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    await asyncio.sleep(backoff_time)
        
        # If primary provider failed and failover is enabled, try alternatives
        if failover:
            backup_providers = [p for p in self.available_providers if p != config.provider]
            for provider in backup_providers:
                try:
                    logger.info(f"Trying failover provider: {provider.value}")
                    model = self.get_default_model(provider)
                    metadata["failover_attempt"] = True
                    metadata["failover_provider"] = provider.value
                    metadata["failover_model"] = model
                    
                    result = await self._call_provider(
                        prompt, 
                        provider, 
                        model, 
                        config.temperature, 
                        config.max_tokens,
                        config.system_prompt,
                        config.stop_sequences,
                        config.extra_params,
                        config.timeout,
                        config.stream
                    )
                    metadata["success"] = True
                    metadata["duration"] = time.time() - metadata["start_time"]
                    
                    # Trigger post-request callbacks
                    for callback in self.post_request_callbacks:
                        try:
                            callback(prompt=prompt, result=result, metadata=metadata)
                        except Exception as e:
                            logger.error(f"Error in post-request callback: {str(e)}")
                    
                    # Update metrics
                    self.total_latency += metadata["duration"]
                    
                    return result, metadata
                except Exception as e:
                    logger.warning(f"Failover provider {provider.value} failed: {str(e)}")
        
        # If we get here, all attempts failed
        self.error_count += 1
        metadata["success"] = False
        metadata["duration"] = time.time() - metadata["start_time"]
        error_msg = f"All LLM providers failed to generate a response after {metadata['attempts']} attempts"
        metadata["error"] = error_msg if not last_error else f"{error_msg}. Last error: {str(last_error)}"
        
        logger.error(error_msg)
        return error_msg, metadata

    async def _call_provider(self, 
                           prompt: str, 
                           provider: LLMProvider, 
                           model: str,
                           temperature: float, 
                           max_tokens: int,
                           system_prompt: Optional[str],
                           stop_sequences: List[str],
                           extra_params: Dict[str, Any],
                           timeout: float,
                           stream: bool) -> str:
        """Call the appropriate provider method based on the provider type"""
        try:
            # Create message structure based on whether system prompt is provided
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            if provider == LLMProvider.GROQ:
                if not GROQ_AVAILABLE or not groq_client:
                    raise ProviderError(provider, ValueError("Groq client not available"))
                return await self._query_groq(messages, model, temperature, max_tokens, stop_sequences, extra_params, timeout, stream)
            
            elif provider == LLMProvider.OPENAI:
                if not OPENAI_AVAILABLE or not openai_client:
                    raise ProviderError(provider, ValueError("OpenAI client not available"))
                return await self._query_openai(messages, model, temperature, max_tokens, stop_sequences, extra_params, timeout, stream)
            
            elif provider == LLMProvider.GEMINI:
                if not GEMINI_AVAILABLE:
                    raise ProviderError(provider, ValueError("Gemini client not available"))
                return await self._query_gemini(messages, model, temperature, max_tokens, stop_sequences, extra_params, timeout, stream)
            
            else:
                raise ValueError(f"Unsupported provider: {provider.value}")
        except Exception as e:
            raise ProviderError(provider, e)

    async def _query_groq(self, 
                        messages: List[Dict[str, str]], 
                        model: str, 
                        temperature: float, 
                        max_tokens: int,
                        stop_sequences: List[str],
                        extra_params: Dict[str, Any],
                        timeout: float,
                        stream: bool) -> str:
        """Query the Groq API"""
        try:
            # Build request parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                params["stop"] = stop_sequences
                
            # Add any extra parameters
            params.update(extra_params)
            
            # Create a task with timeout
            completion_task = asyncio.create_task(
                asyncio.to_thread(groq_client.chat.completions.create, **params)
            )
            
            try:
                # Wait for completion with timeout
                completion = await asyncio.wait_for(completion_task, timeout=timeout)
                return completion.choices[0].message.content
            except asyncio.TimeoutError:
                # Cancel the task if it times out
                completion_task.cancel()
                try:
                    await completion_task
                except asyncio.CancelledError:
                    pass
                raise TimeoutError(f"Request to Groq API timed out after {timeout} seconds")
                
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    async def _query_openai(self, 
                          messages: List[Dict[str, str]], 
                          model: str, 
                          temperature: float, 
                          max_tokens: int,
                          stop_sequences: List[str],
                          extra_params: Dict[str, Any],
                          timeout: float,
                          stream: bool) -> str:
        """Query the OpenAI API"""
        try:
            # Build request parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                params["stop"] = stop_sequences
                
            # Add any extra parameters
            params.update(extra_params)
            
            # Create a task with timeout
            completion_task = asyncio.create_task(
                asyncio.to_thread(openai_client.chat.completions.create, **params)
            )
            
            try:
                # Wait for completion with timeout
                completion = await asyncio.wait_for(completion_task, timeout=timeout)
                return completion.choices[0].message.content
            except asyncio.TimeoutError:
                # Cancel the task if it times out
                completion_task.cancel()
                try:
                    await completion_task
                except asyncio.CancelledError:
                    pass
                raise TimeoutError(f"Request to OpenAI API timed out after {timeout} seconds")
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _query_gemini(self,
                          messages: List[Dict[str, str]],
                          model: str,
                          temperature: float,
                          max_tokens: int,
                          stop_sequences: List[str],
                          extra_params: Dict[str, Any],
                          timeout: float,
                          stream: bool) -> str:
        """Query the Gemini API"""
        try:
            # Extract system prompt if present and user content
            system_prompt = None
            content_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    content_parts.append(msg["content"])

            # Combine user messages into a single prompt
            prompt = "\n\n".join(content_parts).strip()
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"

            if not prompt:
                return ""

            # Instantiate the Gemini model by name
            gemini_model = genai.GenerativeModel(model)

            # Perform synchronous call in a thread with timeout
            task = asyncio.create_task(
                asyncio.to_thread(gemini_model.generate_content, prompt)
            )
            try:
                response = await asyncio.wait_for(task, timeout=timeout)
                return response.text
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise TimeoutError(f"Request to Gemini API timed out after {timeout} seconds")

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for the LLM handler"""
        if self.request_count > 0:
            avg_latency = self.total_latency / self.request_count
            error_rate = (self.error_count / self.request_count) * 100
        else:
            avg_latency = 0.0
            error_rate = 0.0
            
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "average_latency": avg_latency,
            "available_providers": [p.value for p in self.available_providers]
        }
        
    def reset_metrics(self):
        """Reset metric counters"""
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0


# Example usage
async def example_usage():
    # Create the handler
    llm_handler = LLMHandler()
    
    # Basic request with defaults
    response, metadata = await llm_handler.generate("Tell me about quantum computing.")
    print(f"Response: {response[:100]}...")
    print(f"Metadata: {metadata}")
    
    # Request with specific configuration
    config = LLMConfig(
        model="gpt-4",
        provider=LLMProvider.OPENAI,
        temperature=0.2,
        max_tokens=500,
        system_prompt="You are a helpful expert in physics."
    )
    
    response, metadata = await llm_handler.generate(
        "Explain the double-slit experiment.", 
        config=config
    )
    
    print(f"Response: {response[:100]}...")
    print(f"Metadata: {metadata}")
    
    # Print metrics
    print(f"Metrics: {llm_handler.get_metrics()}")


# if __name__ == "__main__":
#     asyncio.run(example_usage())

# At bottom of llm.py (for quick local sanity test)
if __name__ == "__main__":
    import asyncio
    handler = LLMHandler()
    print("Providers:", handler.available_providers)
    async def _test():
        resp, meta = await handler.generate("Hello world", LLMConfig(
            model=handler.get_default_model(handler.available_providers[2]),
            provider=handler.available_providers[2],
            max_tokens=10
        ))
        print("Output:", resp)
        print("Meta:", meta)
    asyncio.run(_test())