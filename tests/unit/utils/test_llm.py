import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import asyncio

# Adjust the path to import from the app directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.app.utils.llm import LLMHandler, LLMProvider, LLMConfig
from backend.app.utils.errors import LLMError

class TestLLMHandler(unittest.TestCase):
    """Test cases for the LLMHandler class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Mock the imported classes and modules correctly
        with patch('groq.AsyncGroq'), \
             patch('openai.AsyncOpenAI'), \
             patch('google.generativeai'), \
             patch('backend.app.utils.llm.os'):
            self.llm_handler = LLMHandler(max_retries=2, retry_delay=0.1)
            
            # Save the original methods
            self.original_generate_groq = self.llm_handler._generate_groq
            self.original_generate_openai = self.llm_handler._generate_openai
            self.original_generate_gemini = self.llm_handler._generate_gemini
            
            # Mock the provider methods
            self.llm_handler._generate_groq = AsyncMock(return_value=("Groq response", {"provider": "groq"}))
            self.llm_handler._generate_openai = AsyncMock(return_value=("OpenAI response", {"provider": "openai"}))
            self.llm_handler._generate_gemini = AsyncMock(return_value=("Gemini response", {"provider": "gemini"}))
    
    def test_init(self):
        """Test LLMHandler initialization"""
        self.assertEqual(self.llm_handler.max_retries, 2)
        self.assertEqual(self.llm_handler.retry_delay, 0.1)
        self.assertIsNotNone(self.llm_handler.groq_client)
        self.assertIsNotNone(self.llm_handler.openai_client)
        self.assertIsNotNone(self.llm_handler.model_mappings)
        
    @pytest.mark.asyncio
    async def test_generate_groq(self):
        """Test generating text with Groq"""
        config = LLMConfig(
            model="mixtral-8x7b-32768",
            provider=LLMProvider.GROQ
        )
        result, metadata = await self.llm_handler.generate(
            prompt="Test prompt", 
            config=config
        )
        
        self.assertEqual(result, "Groq response")
        self.llm_handler._generate_groq.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_generate_openai(self):
        """Test generating text with OpenAI"""
        config = LLMConfig(
            model="gpt-4-turbo-preview",
            provider=LLMProvider.OPENAI
        )
        result, metadata = await self.llm_handler.generate(
            prompt="Test prompt",
            config=config
        )
        
        self.assertEqual(result, "OpenAI response")
        self.llm_handler._generate_openai.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_generate_gemini(self):
        """Test generating text with Gemini"""
        config = LLMConfig(
            model="gemini-1.5-flash",
            provider=LLMProvider.GEMINI
        )
        result, metadata = await self.llm_handler.generate(
            prompt="Test prompt",
            config=config
        )
        
        self.assertEqual(result, "Gemini response")
        self.llm_handler._generate_gemini.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_generate_retry(self):
        """Test retrying on failure"""
        # Set first attempt to fail, second to succeed
        self.llm_handler._generate_with_provider = AsyncMock()
        self.llm_handler._generate_with_provider.side_effect = [
            Exception("Temporary failure"),
            ("Retry success", {"provider": "groq"})
        ]
        
        config = LLMConfig(
            model="mixtral-8x7b-32768",
            provider=LLMProvider.GROQ
        )
        
        result, metadata = await self.llm_handler.generate(
            prompt="Test prompt", 
            config=config
        )
        
        self.assertEqual(result, "Retry success")
        self.assertEqual(self.llm_handler._generate_with_provider.call_count, 2)
        
    @pytest.mark.asyncio
    async def test_generate_max_retries_exceeded_and_failover(self):
        """Test exceeding max retries leading to failover"""
        # Make all attempts for primary provider fail
        self.llm_handler._generate_with_provider = AsyncMock()
        self.llm_handler._generate_with_provider.side_effect = Exception("Persistent failure")
        
        # But make failover succeed
        self.llm_handler._failover_generation = AsyncMock(
            return_value=("Failover response", {"provider": "openai"})
        )
        
        config = LLMConfig(
            model="mixtral-8x7b-32768",
            provider=LLMProvider.GROQ
        )
        
        result, metadata = await self.llm_handler.generate(
            prompt="Test prompt",
            config=config
        )
        
        # Should have tried the primary provider max_retries times then used failover
        self.assertEqual(result, "Failover response")
        self.assertEqual(self.llm_handler._generate_with_provider.call_count, 2)  # Initial + 1 retry
        self.llm_handler._failover_generation.assert_called_once()
        
    @pytest.mark.asyncio
    async def test__generate_groq(self):
        """Test _generate_groq method"""
        # Restore the original method
        self.llm_handler._generate_groq = self.original_generate_groq
        
        # Mock the groq client's chat completion method
        self.llm_handler.groq_client.chat.completions.create = AsyncMock()
        self.llm_handler.groq_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="Groq test response"),
                finish_reason="stop"
            )]
        )
        
        config = LLMConfig(
            model="mixtral-8x7b-32768",
            provider=LLMProvider.GROQ,
            temperature=0.7
        )
        
        result, metadata = await self.llm_handler._generate_groq(
            "Test prompt",
            config,
            {}
        )
        
        self.assertEqual(result, "Groq test response")
        self.assertEqual(metadata["provider"], "groq")
        self.llm_handler.groq_client.chat.completions.create.assert_called_once()
        
    @pytest.mark.asyncio
    async def test__generate_openai(self):
        """Test _generate_openai method"""
        # Restore the original method
        self.llm_handler._generate_openai = self.original_generate_openai
        
        # Mock the openai client's chat completion method
        self.llm_handler.openai_client.chat.completions.create = AsyncMock()
        self.llm_handler.openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="OpenAI test response"),
                finish_reason="stop"
            )]
        )
        
        config = LLMConfig(
            model="gpt-4-turbo-preview",
            provider=LLMProvider.OPENAI,
            temperature=0.7
        )
        
        result, metadata = await self.llm_handler._generate_openai(
            "Test prompt",
            config,
            {}
        )
        
        self.assertEqual(result, "OpenAI test response")
        self.assertEqual(metadata["provider"], "openai")
        self.llm_handler.openai_client.chat.completions.create.assert_called_once()
        
    @pytest.mark.asyncio
    async def test__generate_gemini(self):
        """Test _generate_gemini method"""
        # Restore the original method
        self.llm_handler._generate_gemini = self.original_generate_gemini
        
        # Create a mock for genai
        mock_genai = unittest.mock.MagicMock()
        mock_response = unittest.mock.MagicMock()
        mock_response.text = "Gemini test response"
        
        # Mock the asyncio.to_thread function to return our mock response
        with patch('backend.app.utils.llm.asyncio.to_thread', AsyncMock(return_value=mock_response)):
            # Mock the GenerativeModel
            mock_model = unittest.mock.MagicMock()
            mock_model.generate_content = unittest.mock.MagicMock()
            
            # Mock genai.GenerativeModel to return our mock model
            with patch('google.generativeai.GenerativeModel', return_value=mock_model):
                config = LLMConfig(
                    model="gemini-1.5-flash",
                    provider=LLMProvider.GEMINI,
                    temperature=0.7
                )
                
                result, metadata = await self.llm_handler._generate_gemini(
                    "Test prompt",
                    config,
                    {}
                )
                
                self.assertEqual(result, "Gemini test response")
                self.assertEqual(metadata["provider"], "gemini")
        
    @pytest.mark.asyncio
    async def test__get_model_priority(self):
        """Test the _get_model_priority method"""
        # Test getting model priority for various models
        # This tests the private method that helps with model selection during failover
        
        # Test for a model that exists in the mappings
        priority = self.llm_handler._get_model_priority(
            LLMProvider.GROQ, 
            self.llm_handler.model_mappings[LLMProvider.GROQ]["default"]
        )
        self.assertEqual(priority, "default")
        
        # Test for a model that exists in the mappings with a different priority
        priority = self.llm_handler._get_model_priority(
            LLMProvider.OPENAI,
            self.llm_handler.model_mappings[LLMProvider.OPENAI]["fast"]
        )
        self.assertEqual(priority, "fast")
        
        # Test for a model that doesn't exist in the mappings
        priority = self.llm_handler._get_model_priority(
            LLMProvider.GEMINI,
            "non-existent-model"
        )
        self.assertEqual(priority, "default")  # Default fallback
        
if __name__ == "__main__":
    unittest.main()