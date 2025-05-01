#backend/utils/llm.py
import os
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Initialize LLM clients
from groq import Groq
import openai
import google.generativeai as genai

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class LLMHandler:
    def __init__(self):
        self.default_provider = "groq"

    async def generate(self, prompt: str, model: str = None, provider: str = None,
                       temperature: float = 0.7, max_tokens: int = 1000) -> str:
        provider = provider or self.default_provider
        model = model or self.get_model_for_provider(provider)

        try:
            if provider == "groq":
                return await self.query_groq(prompt, model)
            elif provider == "openai":
                return await self.query_openai(prompt, model, temperature, max_tokens)
            elif provider == "gemini":
                return await self.query_gemini(prompt, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"LLMHandler Error (provider={provider}): {e}")
            return f"[{provider.upper()} Error] {str(e)}"

    async def query_groq(self, prompt: str, model: str) -> str:
        try:
            completion = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    async def query_openai(self, prompt: str, model: str,
                           temperature: float, max_tokens: int) -> str:
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def query_gemini(self, prompt: str, model: str) -> str:
        try:
            gemini_model = genai.GenerativeModel(model)
            response = await asyncio.to_thread(gemini_model.generate_content, prompt)
            return getattr(response, "text", str(response))
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def get_model_for_provider(self, provider: str) -> str:
        """Returns default model name based on provider."""
        mapping = {
            "groq": "llama3-70b-8192-versatile",
            "openai": "gpt-4",
            "gemini": "gemini-pro"
        }
        return mapping.get(provider, "gpt-4")