import sys
import os
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# Adjust the path to import from the app directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.app.mcp.mcp_server import MCPServer
from backend.app.agents.agent_core import AgentManager
from backend.app.utils.memory_manager import MemoryManager

class TestMCPServer(unittest.TestCase):
    """Test cases for the MCPServer class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create test instance with mocked dependencies
        with patch('backend.app.mcp.mcp_server.LLMHandler'), \
             patch('backend.app.mcp.mcp_server.AgentManager'), \
             patch('backend.app.mcp.mcp_server.MemoryManager'), \
             patch('backend.app.mcp.mcp_server.GuardrailsChecker'), \
             patch('backend.app.mcp.mcp_server.IntentDetectorAgent'), \
             patch('backend.app.mcp.mcp_server.RetrieverAgent'), \
             patch('backend.app.mcp.mcp_server.ResearcherAgent'), \
             patch('backend.app.mcp.mcp_server.VerifierAgent'), \
             patch('backend.app.mcp.mcp_server.SummarizerAgent'):
                self.mcp_server = MCPServer(model="test-model", provider="test-provider")
                
                # Mock all agents for testing
                self.mcp_server.intent_detector = MagicMock()
                self.mcp_server.intent_detector.detect_intent = AsyncMock()
                self.mcp_server.intent_detector.run = AsyncMock()
                self.mcp_server.intent_detector.__class__.__name__ = "IntentDetectorAgent"
                
                self.mcp_server.retriever_agent = MagicMock()
                self.mcp_server.retriever_agent.run = AsyncMock()
                self.mcp_server.retriever_agent.__class__.__name__ = "RetrieverAgent"
                self.mcp_server.retriever_agent.agent_id = "retriever-agent-id"
                
                self.mcp_server.researcher_agent = MagicMock()
                self.mcp_server.researcher_agent.run = AsyncMock()
                self.mcp_server.researcher_agent.__class__.__name__ = "ResearcherAgent"
                self.mcp_server.researcher_agent.agent_id = "researcher-agent-id"
                
                self.mcp_server.verifier_agent = MagicMock()
                self.mcp_server.verifier_agent.run = AsyncMock()
                self.mcp_server.verifier_agent.__class__.__name__ = "VerifierAgent"
                self.mcp_server.verifier_agent.agent_id = "verifier-agent-id"
                
                self.mcp_server.summarizer_agent = MagicMock()
                self.mcp_server.summarizer_agent.run = AsyncMock()
                self.mcp_server.summarizer_agent.__class__.__name__ = "SummarizerAgent"
                self.mcp_server.summarizer_agent.agent_id = "summarizer-agent-id"
                
                # Mock agent manager
                self.mcp_server.agent_manager = MagicMock()
                self.mcp_server.agent_manager.get_agent_by_id = MagicMock()
                self.mcp_server.agent_manager.set_active_agent = MagicMock()
                
                # Mock memory manager
                self.mcp_server.memory_manager = MagicMock()
                self.mcp_server.memory_manager.create_session = MagicMock(return_value="test-session-id")
                self.mcp_server.memory_manager.save_session = MagicMock()
                self.mcp_server.memory_manager.load_session = MagicMock()
                self.mcp_server.memory_manager.get_timestamp = MagicMock(return_value="2025-05-05T12:00:00")
                
                # Mock guardrails
                self.mcp_server.guardrails = MagicMock()
                self.mcp_server.guardrails.sanitize_output = MagicMock(side_effect=lambda x: x)
                
                # Initialize attributes
                self.mcp_server.handoff_history = []
                self.mcp_server.trace = []
                self.mcp_server.global_context = {}
                self.mcp_server.session_id = None

    def test_init(self):
        """Test MCP Server initialization"""
        self.assertEqual(self.mcp_server.model, "test-model")
        self.assertEqual(self.mcp_server.provider, "test-provider")
        self.assertIsInstance(self.mcp_server.trace, list)
        self.assertIsNone(self.mcp_server.session_id)
        self.assertIsInstance(self.mcp_server.global_context, dict)
        
    def test_update_model_settings(self):
        """Test updating model settings"""
        # Test with valid provider and model
        self.mcp_server.available_models = {
            "groq": ["llama3-8b-8192"],
            "anthropic": ["claude-3-opus-20240229"]
        }
        
        # Mock the _initialize_agents method to prevent actual agent initialization
        self.mcp_server._initialize_agents = MagicMock()
        
        # Test with valid provider and model
        self.mcp_server.update_model_settings("claude-3-opus-20240229", "anthropic")
        self.assertEqual(self.mcp_server.model, "claude-3-opus-20240229")
        self.assertEqual(self.mcp_server.provider, "anthropic")
        
        # Test with invalid provider
        self.mcp_server.update_model_settings("any-model", "invalid-provider")
        self.assertEqual(self.mcp_server.provider, "groq")
        
        # Test with invalid model
        self.mcp_server.update_model_settings("invalid-model", "groq")
        self.assertEqual(self.mcp_server.model, "llama3-8b-8192")

    def test_get_fallback_provider(self):
        """Test fallback provider logic"""
        # Set up test data
        self.mcp_server.fallback_sequence = {
            "test-provider": ["fallback1", "fallback2"]
        }
        self.mcp_server.available_models = {
            "fallback1": ["fallback1-model"],
            "fallback2": ["fallback2-model"],
            "groq": ["llama3-8b-8192"]
        }
    
        # Test with provider that has fallbacks
        provider, model = self.mcp_server.get_fallback_provider("test-provider")
        self.assertEqual(provider, "fallback1")
        self.assertEqual(model, "fallback1-model")
    
        # Test with provider that doesn't have specific fallbacks
        provider, model = self.mcp_server.get_fallback_provider("unknown-provider")
        # Actual implementation uses 'groq' as default fallback instead of 'openai'
        self.assertEqual(provider, "groq")  # Default fallback provider

    @pytest.mark.asyncio
    async def test_start_session(self):
        """Test session start/resume logic"""
        # Test creating new session
        session_id = await self.mcp_server.start_session()
        self.assertEqual(session_id, "test-session-id")
        self.assertEqual(self.mcp_server.session_id, "test-session-id")
        
        # Test resuming session
        self.mcp_server.memory_manager.load_session.return_value = {
            "global_context": {"test_key": "test_value"}
        }
        session_id = await self.mcp_server.start_session("existing-session")
        self.assertEqual(session_id, "existing-session")
        self.assertEqual(self.mcp_server.session_id, "existing-session")
        self.assertEqual(self.mcp_server.global_context, {"test_key": "test_value"})
        
        # Test session with error
        self.mcp_server.memory_manager.create_session.side_effect = Exception("Test error")
        session_id = await self.mcp_server.start_session()
        self.assertTrue(session_id.startswith("fallback_"))

    @pytest.mark.asyncio
    async def test_agent_handoff(self):
        """Test agent handoff functionality"""
        # Mock get_agent_by_id to return the actual mock agents
        self.mcp_server.agent_manager.get_agent_by_id.side_effect = lambda agent_id: {
            "researcher-agent-id": self.mcp_server.researcher_agent,
            "retriever-agent-id": self.mcp_server.retriever_agent,
            "summarizer-agent-id": self.mcp_server.summarizer_agent,
            "verifier-agent-id": self.mcp_server.verifier_agent
        }.get(agent_id)
        
        # Set up return values for agent runs
        self.mcp_server.summarizer_agent.run.return_value = "Summarized response"
        
        # Test handoff
        result = await self.mcp_server.agent_handoff(
            from_agent_id="researcher-agent-id",
            to_agent_id="summarizer-agent-id",
            reason="Need to summarize research findings",
            user_input="Tell me about quantum computing",
            context={"additional": "context"}
        )
        
        # Verify the handoff was recorded
        self.assertEqual(len(self.mcp_server.handoff_history), 1)
        self.assertEqual(self.mcp_server.handoff_history[0]["from_agent"], "ResearcherAgent")
        self.assertEqual(self.mcp_server.handoff_history[0]["to_agent"], "SummarizerAgent")
        self.assertEqual(self.mcp_server.handoff_history[0]["reason"], "Need to summarize research findings")
        
        # Verify the correct agent was made active
        self.mcp_server.agent_manager.set_active_agent.assert_called_with("summarizer-agent-id")
        
        # Verify the receiving agent was called with the enhanced prompt
        self.assertTrue(self.mcp_server.summarizer_agent.run.called)
        
        # Verify the response
        self.assertEqual(result["agent"], "SummarizerAgent")
        self.assertEqual(result["response"], "Summarized response")

    @pytest.mark.asyncio
    async def test_direct_query_workflow(self):
        """Test direct query workflow"""
        self.mcp_server.summarizer_agent.run.return_value = "Brief answer"
        
        result = await self.mcp_server.direct_query_workflow("What is Python?")
        
        self.assertEqual(result, "Brief answer")
        self.assertEqual(len(self.mcp_server.trace), 1)
        self.assertEqual(self.mcp_server.trace[0]["agent"], "SummarizerAgent")
        self.assertTrue(self.mcp_server.summarizer_agent.run.called)

    @pytest.mark.asyncio
    async def test_summary_workflow(self):
        """Test summary workflow"""
        self.mcp_server.retriever_agent.run.return_value = "Retrieved context"
        self.mcp_server.summarizer_agent.run.return_value = "Detailed answer"
        
        result = await self.mcp_server.summary_workflow("Explain quantum computing")
        
        self.assertEqual(result, "Detailed answer")
        self.assertEqual(len(self.mcp_server.trace), 2)
        self.assertEqual(self.mcp_server.trace[0]["agent"], "RetrieverAgent")
        self.assertEqual(self.mcp_server.trace[1]["agent"], "SummarizerAgent")
        self.assertTrue(self.mcp_server.retriever_agent.run.called)
        self.assertTrue(self.mcp_server.summarizer_agent.run.called)

    @pytest.mark.asyncio
    async def test_research_workflow(self):
        """Test research workflow"""
        self.mcp_server.retriever_agent.run.return_value = "Research papers and context"
        self.mcp_server.researcher_agent.run.return_value = "Research analysis"
        self.mcp_server.verifier_agent.run.return_value = "Verified research analysis"
        
        result = await self.mcp_server.research_workflow("Recent advances in quantum computing")
        
        self.assertEqual(result, "Verified research analysis")
        self.assertEqual(len(self.mcp_server.trace), 3)
        self.assertEqual(self.mcp_server.trace[0]["agent"], "RetrieverAgent")
        self.assertEqual(self.mcp_server.trace[1]["agent"], "ResearcherAgent")
        self.assertEqual(self.mcp_server.trace[2]["agent"], "VerifierAgent")

    @pytest.mark.asyncio
    async def test_research_workflow_verification_failure(self):
        """Test research workflow when verification fails"""
        self.mcp_server.retriever_agent.run.return_value = "Research papers and context"
        self.mcp_server.researcher_agent.run.return_value = "Research analysis"
        self.mcp_server.verifier_agent.run.return_value = "Error: verification failed"
        
        result = await self.mcp_server.research_workflow("Recent advances in quantum computing")
        
        # Should return original research response when verification fails
        self.assertEqual(result, "Research analysis")

    @pytest.mark.asyncio
    async def test_run_with_fallback_success(self):
        """Test successful execution without fallback"""
        mock_func = AsyncMock(return_value="Success")
        
        result = await self.mcp_server._run_with_fallback(mock_func, "test_arg")
        
        self.assertEqual(result, "Success")
        mock_func.assert_called_once_with("test_arg")
        self.assertEqual(len(self.mcp_server.trace), 0)  # No fallback, no trace entry

    @pytest.mark.asyncio
    async def test_run_with_fallback_failure(self):
        """Test execution with fallback mechanism"""
        # Mock function that fails on first attempt but succeeds on fallback
        mock_func = AsyncMock(side_effect=[Exception("Test failure"), "Fallback success"])
        
        # Configure fallback provider
        self.mcp_server.get_fallback_provider = MagicMock(return_value=("fallback-provider", "fallback-model"))
        self.mcp_server.update_model_settings = MagicMock()
        
        result = await self.mcp_server._run_with_fallback(mock_func, "test_arg")
        
        self.assertEqual(result, "Fallback success")
        self.assertEqual(mock_func.call_count, 2)
        self.assertEqual(len(self.mcp_server.trace), 1)
        self.assertTrue(self.mcp_server.trace[0]["fallback"])

    @pytest.mark.asyncio
    async def test_run_with_fallback_all_fail(self):
        """Test execution when all fallback attempts fail"""
        # Mock function that fails on all attempts
        mock_func = AsyncMock(side_effect=Exception("Test failure"))
        
        # Configure fallback provider
        self.mcp_server.get_fallback_provider = MagicMock(return_value=("fallback-provider", "fallback-model"))
        self.mcp_server.update_model_settings = MagicMock()
        
        result = await self.mcp_server._run_with_fallback(mock_func, "test_arg")
        
        self.assertIsNone(result)
        self.assertTrue(mock_func.call_count >= 1)
        self.assertEqual(len(self.mcp_server.trace), 1)
        self.assertTrue(self.mcp_server.trace[0]["fallback_failed"])

    @pytest.mark.asyncio
    async def test_route_simple_responses(self):
        """Test simple conversational responses"""
        result = await self.mcp_server.route("hello")
        self.assertEqual(result["final_output"], "Hello! How can I assist you today?")
        
        result = await self.mcp_server.route("goodbye")
        self.assertEqual(result["final_output"], "Goodbye! Feel free to come back if you have more questions.")
        
        result = await self.mcp_server.route("thank you")
        self.assertEqual(result["final_output"], "You're welcome! 😊")

    @pytest.mark.asyncio
    async def test_route_chat_mode(self):
        """Test routing in chat mode"""
        # Mock intent detection
        self.mcp_server.intent_detector.detect_intent.return_value = {
            "mode": "chat",
            "depth": "brief",
            "domain_valid": True
        }
        
        # Mock direct query workflow
        self.mcp_server._run_with_fallback = AsyncMock(return_value="Brief chat response")
        
        result = await self.mcp_server.route("What is Python?", mode="chat")
        
        self.assertEqual(result["final_output"], "Brief chat response")
        self.assertTrue(any(trace["result"] == "direct_query" for trace in result["trace"]))

    @pytest.mark.asyncio
    async def test_route_research_mode(self):
        """Test routing in research mode"""
        # Mock intent detection
        self.mcp_server.intent_detector.detect_intent.return_value = {
            "mode": "research",
            "depth": "detailed",
            "domain_valid": True,
            "requires_papers": True
        }
        
        # Mock research workflow
        self.mcp_server._run_with_fallback = AsyncMock(return_value="Research response with papers")
        
        result = await self.mcp_server.route(
            "Explain recent advances in quantum computing", 
            mode="research"
        )
        
        self.assertEqual(result["final_output"], "Research response with papers")
        self.assertTrue(any(trace["result"] == "research_with_papers" for trace in result["trace"]))

if __name__ == "__main__":
    unittest.main()