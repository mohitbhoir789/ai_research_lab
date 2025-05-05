import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# Adjust the path to import from the app directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.app.agents.intent_detector_agent import IntentDetectorAgent

class TestIntentDetectorAgent(unittest.TestCase):
    """Test cases for the IntentDetectorAgent class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Instead of patching LLMHandler at import time, create the agent directly
        # and set up the mocks afterward
        self.agent = IntentDetectorAgent(model="test-model", provider="test-provider")
        self.agent.llm = AsyncMock()
        self.agent.llm.generate = AsyncMock()
        
        # Mock GuardrailsChecker
        self.agent.guardrails = MagicMock()
        self.agent.guardrails.validate_user_input = MagicMock(return_value=True)
            
    def test_init(self):
        """Test IntentDetectorAgent initialization"""
        # Check for attributes that should exist
        self.assertEqual(self.agent.model, "test-model")
        self.assertEqual(self.agent.provider, "test-provider")
        
        # Check for agent_type only if it exists
        if hasattr(self.agent, 'agent_type'):
            self.assertEqual(self.agent.agent_type, "intent_detector")
        
    @pytest.mark.asyncio
    async def test_detect_intent_basic(self):
        """Test basic intent detection"""
        # Mock LLM response for a chat intent
        self.agent.llm.generate.return_value = """
{
    "mode": "chat",
    "depth": "brief",
    "domain_valid": true,
    "requires_papers": false,
    "topics": ["python", "programming"],
    "confidence": 0.9
}
"""
        
        result = await self.agent.detect_intent("What is Python?")
        
        self.agent.llm.generate.assert_called_once()
        self.assertEqual(result["mode"], "chat")
        self.assertEqual(result["depth"], "brief")
        self.assertTrue(result["domain_valid"])
        self.assertFalse(result["requires_papers"])
        
    @pytest.mark.asyncio
    async def test_detect_intent_research(self):
        """Test intent detection for research queries"""
        # Mock LLM response for a research intent
        self.agent.llm.generate.return_value = """
{
    "mode": "research",
    "depth": "detailed",
    "domain_valid": true,
    "requires_papers": true,
    "topics": ["machine learning", "deep learning", "neural networks"],
    "confidence": 0.95
}
"""
        
        result = await self.agent.detect_intent("Explain recent advances in deep learning techniques")
        
        self.agent.llm.generate.assert_called_once()
        self.assertEqual(result["mode"], "research")
        self.assertEqual(result["depth"], "detailed")
        self.assertTrue(result["domain_valid"])
        self.assertTrue(result["requires_papers"])
        
    @pytest.mark.asyncio
    async def test_detect_intent_invalid_domain(self):
        """Test intent detection for queries outside allowed domains"""
        # Mock LLM response for an invalid domain
        self.agent.llm.generate.return_value = """
{
    "mode": "chat",
    "depth": "brief",
    "domain_valid": false,
    "requires_papers": false,
    "topics": ["politics"],
    "confidence": 0.8
}
"""
        
        # Mock guardrails validation to fail
        self.agent.guardrails.validate_user_input = MagicMock(return_value=False)
        
        result = await self.agent.detect_intent("What's your opinion on current politics?")
        
        self.agent.llm.generate.assert_called_once()
        self.assertFalse(result["domain_valid"])
        
    @pytest.mark.asyncio
    async def test_detect_intent_json_error(self):
        """Test intent detection with invalid JSON response"""
        # Mock LLM response with invalid JSON
        self.agent.llm.generate.return_value = "This is not valid JSON"
        
        result = await self.agent.detect_intent("What is Python?")
        
        self.agent.llm.generate.assert_called_once()
        self.assertEqual(result["mode"], "error")
        self.assertFalse(result["domain_valid"])
        
    @pytest.mark.asyncio
    async def test_detect_intent_llm_error(self):
        """Test intent detection when LLM raises an error"""
        # Mock LLM to raise an exception
        self.agent.llm.generate.side_effect = Exception("LLM error")
        
        result = await self.agent.detect_intent("What is Python?")
        
        self.agent.llm.generate.assert_called_once()
        self.assertEqual(result["mode"], "error")
        self.assertFalse(result["domain_valid"])
        self.assertEqual(self.agent.last_error, "LLM error")
        
    @pytest.mark.asyncio
    async def test_run(self):
        """Test run method"""
        # Mock detect_intent method
        self.agent.detect_intent = AsyncMock(return_value={
            "mode": "chat",
            "depth": "brief",
            "domain_valid": True
        })
        
        result = await self.agent.run("What is Python?")
        
        self.agent.detect_intent.assert_called_once_with("What is Python?")
        self.assertIn("Intent detected:", result)
        self.assertIn("mode: chat", result)

if __name__ == "__main__":
    unittest.main()