"""
MCPServer Module
Main orchestrator for the AI Research Assistant application.
Handles routing between different specialized agents.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio

# Fix import path consistency
from app.agents.agent_core import AgentManager, BaseAgent, LLMAgent, Memory, Tool
from app.agents.retriever_agent import RetrieverAgent
from app.agents.researcher_agent import ResearcherAgent
from app.agents.verifier_agent import VerifierAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.agents.intent_detector_agent import IntentDetectorAgent
from app.utils.llm import LLMHandler, LLMProvider, LLMConfig
from app.utils.memory_manager import MemoryManager
from app.utils.guardrails import GuardrailsChecker
from app.utils.errors import LLMError

logger = logging.getLogger(__name__)

class MCPServer:
    """
    Master Control Program Server - Orchestrates all agent interactions
    """

    def __init__(self, model="llama3-8b-8192", provider="groq"):
        """Initialize the MCP server with all specialized agents"""
        # Initialize LLM handler for utility functions
        self.llm = LLMHandler(max_retries=3, retry_delay=1.0)

        # Initialize agent manager
        self.agent_manager = AgentManager()

        # Initialize memory manager for persistent memory across sessions
        self.memory_manager = MemoryManager()

        # Initialize guardrails checker
        self.guardrails = GuardrailsChecker()

        # Default model and provider (updated with currently available Groq models)
        self.model = model
        self.provider = provider
        
        # Fallback configuration
        self.available_models = {
            "groq": ["llama3-8b-8192", "llama3-70b-8192", "gemma-7b-it", "mixtral-8x7b-32768", 
                    "llama2-70b-4096"],
            "gemini": ["gemini-1.5-flash", "gemini-2.0-flash"],
            "openai": ["gpt-4-turbo", "gpt-3.5-turbo"]
        }
        
        # Fallback sequences for each provider
        self.fallback_sequence = {
            "groq": ["openai", "gemini"],
            "openai": ["groq", "gemini"],
            "gemini": ["openai", "groq"]
        }

        # Initialize all specialized agents
        self._initialize_agents()

        # Trace for monitoring and debugging
        self.trace: List[Dict[str, Any]] = []

        # Session ID for tracking conversation context
        self.session_id = None

        # Global context shared between agents
        self.global_context = {}

    def _initialize_agents(self):
        """Initialize and register all specialized agents"""
        try:
            # Initialize specialized agents with the current model and provider
            self.intent_detector = IntentDetectorAgent(model=self.model, provider=self.provider)
            self.retriever_agent = RetrieverAgent(model=self.model, provider=self.provider)
            self.researcher_agent = ResearcherAgent(model=self.model, provider=self.provider)
            self.verifier_agent = VerifierAgent(model=self.model, provider=self.provider)
            self.summarizer_agent = SummarizerAgent(model=self.model, provider=self.provider)

            # Register all agents with the agent manager
            self.agent_manager.register_agent(self.intent_detector)
            self.agent_manager.register_agent(self.retriever_agent)
            self.agent_manager.register_agent(self.researcher_agent)
            self.agent_manager.register_agent(self.verifier_agent)
            self.agent_manager.register_agent(self.summarizer_agent)

            # Set the researcher agent as the default active agent
            self.agent_manager.set_active_agent(self.researcher_agent.agent_id)
            
            logger.info("Successfully initialized and registered all agents")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise

    def update_model_settings(self, model: str, provider: str):
        """
        Update the model and provider for all agents

        Args:
            model: The LLM model to use
            provider: The LLM provider (groq, openai, gemini)
        """
        # Validate provider and model
        if provider not in self.available_models:
            logger.warning(f"Unsupported provider: {provider}, falling back to groq")
            provider = "groq"
            
        if provider in self.available_models and model not in self.available_models[provider]:
            logger.warning(f"Model {model} not available for provider {provider}, using default")
            model = self.available_models[provider][0]
        
        self.model = model
        self.provider = provider

        # Re-initialize all agents with new settings
        self._initialize_agents()

        logger.info(f"Updated model settings: model={model}, provider={provider}")
        
    def get_fallback_provider(self, current_provider: str) -> Tuple[str, str]:
        """
        Get fallback provider and model when the current provider fails
        
        Args:
            current_provider: The provider that failed
            
        Returns:
            Tuple of (provider, model)
        """
        # Get fallback sequence for the current provider
        fallback_sequence = self.fallback_sequence.get(current_provider, ["openai", "groq", "gemini"])
        
        # Try each provider in the fallback sequence
        for provider in fallback_sequence:
            if provider in self.available_models and self.available_models[provider]:
                # Return the first available model for this provider
                return provider, self.available_models[provider][0]
        
        # If all fallbacks fail, return the default
        return "groq", "llama3-8b-8192"

    async def _run_with_fallback(self, func, *args, **kwargs):
        """
        Run a function with fallback mechanisms if it fails
        
        Args:
            func: The async function to run
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function or None if all attempts fail
        """
        original_provider = self.provider
        original_model = self.model
        
        # First try with the current provider/model
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(f"Error with provider {self.provider}: {str(e)}")
            self.trace.append({
                "stage": "fallback",
                "error": str(e),
                "original_provider": original_provider,
                "fallback": True
            })
            
            # Get fallback provider and model
            fallback_provider, fallback_model = self.get_fallback_provider(self.provider)
            
            # Try with fallback provider
            try:
                # Update model settings to use fallback
                self.update_model_settings(fallback_model, fallback_provider)
                
                # Try again with the new model
                result = await func(*args, **kwargs)
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                self.trace.append({
                    "stage": "fallback",
                    "error": str(fallback_error),
                    "fallback_provider": fallback_provider,
                    "fallback_failed": True
                })
                
                # Restore original settings
                self.update_model_settings(original_model, original_provider)
                return None

    async def direct_query_workflow(self, user_input: str) -> str:
        """
        Simple workflow for brief answers and direct queries
        
        Args:
            user_input: The user's query
            
        Returns:
            Response from the summarizer agent
        """
        try:
            # Use summarizer agent for direct, brief responses
            response = await self.summarizer_agent.run(user_input)
            
            # Add to trace for tracking
            self.trace.append({
                "agent": self.summarizer_agent.__class__.__name__,
                "stage": "direct_response",
                "success": True
            })
            
            return response
        except Exception as e:
            logger.error(f"Error in direct_query_workflow: {str(e)}")
            self.trace.append({
                "agent": self.summarizer_agent.__class__.__name__,
                "stage": "direct_response",
                "success": False,
                "error": str(e)
            })
            raise
            
    async def summary_workflow(self, user_input: str) -> str:
        """
        Two-step workflow with retrieval and summarization for detailed answers
        
        Args:
            user_input: The user's query
            
        Returns:
            Response from the summarizer agent with retrieved context
        """
        try:
            # Step 1: Get relevant context from retriever agent
            retrieved_context = await self.retriever_agent.run(user_input)
            
            self.trace.append({
                "agent": self.retriever_agent.__class__.__name__,
                "stage": "retrieval",
                "success": True
            })
            
            # Step 2: Summarize with context
            enhanced_prompt = f"""
            QUERY: {user_input}
            
            CONTEXT: {retrieved_context}
            
            Based on the above context, provide a detailed answer to the query.
            """
            
            response = await self.summarizer_agent.run(enhanced_prompt)
            
            self.trace.append({
                "agent": self.summarizer_agent.__class__.__name__,
                "stage": "summarization",
                "success": True
            })
            
            return response
        except Exception as e:
            logger.error(f"Error in summary_workflow: {str(e)}")
            self.trace.append({
                "stage": "summary_workflow",
                "success": False,
                "error": str(e)
            })
            raise
            
    async def research_workflow(self, user_input: str) -> str:
        """
        Advanced workflow with research paper analysis, verification, and summarization
        
        Args:
            user_input: The user's query
            
        Returns:
            Verified and summarized response
        """
        try:
            # Step 1: Get relevant context from retriever agent but limit size
            retrieved_context = await self.retriever_agent.run(user_input)
            
            # Truncate context if it's too long (max 1500 chars)
            if len(retrieved_context) > 1500:
                retrieved_context = retrieved_context[:1500] + "...[truncated for brevity]"
            
            self.trace.append({
                "agent": self.retriever_agent.__class__.__name__,
                "stage": "retrieval",
                "success": True
            })
            
            # Step 2: Research with retrieved context - use concise prompt structure
            enhanced_prompt = f"""RESEARCH QUESTION: {user_input}
CONTEXT: {retrieved_context}
Provide concise research analysis on the question based on the context."""
            
            research_response = await self.researcher_agent.run(enhanced_prompt)
            
            self.trace.append({
                "agent": self.researcher_agent.__class__.__name__,
                "stage": "research",
                "success": True
            })
            
            # Skip verification step if research response is too long
            if len(research_response) > 2000:
                logger.info("Research response too long, skipping verification")
                return research_response
                
            # Step 3: Verify the research findings with compact prompt
            verification_prompt = f"""QUESTION: {user_input}
RESEARCH: {research_response}
Verify key claims briefly."""
            
            verification_result = await self.verifier_agent.run(verification_prompt)
            
            self.trace.append({
                "agent": self.verifier_agent.__class__.__name__,
                "stage": "verification",
                "success": True
            })
            
            # Only use verification result if it's not an error
            if verification_result.lower().startswith("error:"):
                logger.warning("Verification failed, using original research response")
                return research_response
                
            return verification_result
        except Exception as e:
            logger.error(f"Error in research_workflow: {str(e)}")
            self.trace.append({
                "stage": "research_workflow",
                "success": False,
                "error": str(e)
            })
            raise

    async def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new session or resume an existing one

        Args:
            session_id: Optional existing session ID to resume

        Returns:
            Session ID (new or existing)
        """
        try:
            if session_id:
                # Attempt to restore session state
                session_data = self.memory_manager.load_session(session_id)
                if session_data:
                    self.global_context = session_data.get("global_context", {})
                    self.session_id = session_id
                    logger.info(f"Resumed existing session: {session_id}")
                else:
                    # Create new session if the provided ID doesn't exist
                    self.session_id = self.memory_manager.create_session()
                    self.global_context = {}
                    logger.info(f"Created new session (invalid ID provided): {self.session_id}")
            else:
                # Create new session
                self.session_id = self.memory_manager.create_session()
                self.global_context = {}
                logger.info(f"Created new session: {self.session_id}")

            return self.session_id
        except Exception as e:
            logger.error(f"Error in start_session: {str(e)}")
            # Fallback to a new session ID in case of error
            self.session_id = f"fallback_{str(int(time.time()))}"
            self.global_context = {}
            return self.session_id

    async def route(self, user_input: str, session_id: Optional[str] = None, mode: str = "chat", uploaded_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main entry point for routing user requests to appropriate workflows

        Args:
            user_input: The user's query or request
            session_id: Optional session ID to maintain context
            mode: The current interface mode ("chat" or "research")
            uploaded_files: Optional list of uploaded file paths

        Returns:
            Dict containing trace and final output
        """
        try:
            # Enforce 1000 character limit on user input to prevent the "query too long" error
            if len(user_input) > 1000:
                user_input = user_input[:1000] + "..."
                logger.warning("User input truncated to 1000 characters")
                
            # Start or resume session
            if session_id != self.session_id:
                await self.start_session(session_id)

            # Reset trace for new request
            self.trace = []
            
            # Ensure we have a conversation history array
            if "conversation_history" not in self.global_context:
                self.global_context["conversation_history"] = []
                
            # Handle simple conversational replies
            lower_input = user_input.lower().strip()
            if lower_input in ["hi", "hello", "hey"]:
                # Add the greeting to conversation history
                self.global_context["conversation_history"].append({
                    "role": "user",
                    "content": user_input[:200] + ("..." if len(user_input) > 200 else ""),  # Only store preview of message
                    "timestamp": self.memory_manager.get_timestamp(),
                })
                
                # Add the response to conversation history
                greeting_response = "Hello! How can I assist you today?"
                self.global_context["conversation_history"].append({
                    "role": "assistant",
                    "content": greeting_response[:100] + ("..." if len(greeting_response) > 100 else ""),
                    "timestamp": self.memory_manager.get_timestamp(),
                })
                
                # Save session state
                self.memory_manager.save_session(self.session_id, {
                    "global_context": self.global_context,
                    "last_trace": [{"stage": "conversation", "result": "greeting"}]
                })
                
                return {
                    "trace": [{"stage": "conversation", "result": "greeting"}], 
                    "final_output": greeting_response,
                    "session_id": self.session_id
                }
            elif lower_input in ["bye", "goodbye"]:
                # Add the farewell to conversation history
                self.global_context["conversation_history"].append({
                    "role": "user",
                    "content": user_input[:200] + ("..." if len(user_input) > 200 else ""),
                    "timestamp": self.memory_manager.get_timestamp(),
                })
                
                # Add the response to conversation history
                farewell_response = "Goodbye! Feel free to come back if you have more questions."
                self.global_context["conversation_history"].append({
                    "role": "assistant",
                    "content": farewell_response[:100] + ("..." if len(farewell_response) > 100 else ""),
                    "timestamp": self.memory_manager.get_timestamp(),
                })
                
                # Save session state
                self.memory_manager.save_session(self.session_id, {
                    "global_context": self.global_context,
                    "last_trace": [{"stage": "conversation", "result": "farewell"}]
                })
                
                return {
                    "trace": [{"stage": "conversation", "result": "farewell"}], 
                    "final_output": farewell_response,
                    "session_id": self.session_id
                }
            elif lower_input in ["thanks", "thank you"]:
                # Add the thanks to conversation history
                self.global_context["conversation_history"].append({
                    "role": "user",
                    "content": user_input[:200] + ("..." if len(user_input) > 200 else ""),
                    "timestamp": self.memory_manager.get_timestamp(),
                })
                
                # Add the response to conversation history
                gratitude_response = "You're welcome! 😊"
                self.global_context["conversation_history"].append({
                    "role": "assistant",
                    "content": gratitude_response[:100] + ("..." if len(gratitude_response) > 100 else ""),
                    "timestamp": self.memory_manager.get_timestamp(),
                })
                
                # Save session state
                self.memory_manager.save_session(self.session_id, {
                    "global_context": self.global_context,
                    "last_trace": [{"stage": "conversation", "result": "gratitude"}]
                })
                
                return {
                    "trace": [{"stage": "conversation", "result": "gratitude"}], 
                    "final_output": gratitude_response,
                    "session_id": self.session_id
                }

            # Detect intent and validate against guardrails using the IntentDetectorAgent
            # Ensure we're passing a trimmed user_input to prevent "query too long" errors
            intent_data = await self._run_with_fallback(
                self.intent_detector.detect_intent,
                user_input
            )
            
            if intent_data is None:
                # If all fallbacks failed, return a generic error response
                return {
                    "trace": [{"stage": "error", "error": "Intent detection failed with all providers"}],
                    "final_output": "I'm having trouble understanding your request. Please try again later.",
                    "session_id": self.session_id
                }
            
            # Add intent detection to trace
            self.trace.append({
                "stage": "intent_detection",
                "result": intent_data.get("mode", "unknown"),
                "depth": intent_data.get("depth", "unknown"),
                "domain_valid": intent_data.get("domain_valid", False)
            })
            
            # Handle invalid domains or guardrail failures
            if not intent_data.get("domain_valid", False) or intent_data.get("mode") == "error":
                return {
                    "trace": self.trace,
                    "final_output": f"Error: I can only assist with Computer Science and Data Science topics. Please rephrase your question to focus on these domains",
                    "session_id": self.session_id
                }

            # Add user input to global context - only store minimal metadata and intent
            # Avoid storing full conversation history to reduce context size
            self.global_context["latest_query"] = {
                "content": user_input[:200],  # Only store first 200 chars of the query
                "timestamp": self.memory_manager.get_timestamp(),
                "mode": mode,
                "intent_type": intent_data.get("mode", "unknown")
            }
            
            # Limit previous queries history size (only store user queries, not full responses)
            # Enforce stricter limits to ensure we stay within the 1000 character total limit
            max_queries_to_store = 2  # Reduced from 3 to 2 to save space
            if len(self.global_context["conversation_history"]) > max_queries_to_store * 2:  # Multiply by 2 since we store user and minimal response metadata
                # Keep only the most recent messages (cutting from the beginning)
                self.global_context["conversation_history"] = self.global_context["conversation_history"][-(max_queries_to_store*2):]
                
            # Add user message to history with shorter snippets
            self.global_context["conversation_history"].append({
                "role": "user",
                "content": user_input[:150] + ("..." if len(user_input) > 150 else ""),  # Reduced from 200 to 150 chars
                "timestamp": self.memory_manager.get_timestamp(),
            })

            # Add uploaded files to context if provided
            if uploaded_files:
                self.global_context["uploaded_files"] = uploaded_files

            # Route based on mode and detected intent
            output = None
            
            if mode == "chat":
                if intent_data.get("depth") == "brief":
                    # For brief answers in chat mode, use direct query workflow
                    self.trace.append({"stage": "workflow", "result": "direct_query"})
                    output = await self._run_with_fallback(
                        self.direct_query_workflow,
                        user_input
                    )
                else:
                    # For detailed answers in chat mode, use summary workflow
                    self.trace.append({"stage": "workflow", "result": "summary"})
                    output = await self._run_with_fallback(
                        self.summary_workflow,
                        user_input
                    )
            else:  # Research mode
                if intent_data.get("requires_papers", False):
                    # Enhanced research workflow with paper retrieval
                    self.trace.append({"stage": "workflow", "result": "research_with_papers"})
                    output = await self._run_with_fallback(
                        self.research_workflow,
                        user_input
                    )
                else:
                    # Basic research without extensive paper retrieval
                    self.trace.append({"stage": "workflow", "result": "research_basic"})
                    output = await self._run_with_fallback(
                        self.researcher_agent.run,
                        user_input
                    )
            
            # If all workflows failed, return a generic response
            if output is None:
                return {
                    "trace": self.trace + [{"stage": "error", "error": "All workflow attempts failed"}],
                    "final_output": "I'm having trouble processing your request. Please try again later.",
                    "session_id": self.session_id
                }

            # Apply guardrails to output
            output = self.guardrails.sanitize_output(output)

            # Add minimal record of the response to the conversation history
            # Only store a title/summary of the response, not the full text
            # Using even shorter snippets to preserve character budget
            self.global_context["conversation_history"].append({
                "role": "assistant",
                "content": output[:30] + ("..." if len(output) > 30 else ""),  # Reduced from 50 to 30 chars
                "timestamp": self.memory_manager.get_timestamp(),
            })

            # Save session state
            self.memory_manager.save_session(self.session_id, {
                "global_context": self.global_context,
                "last_trace": self.trace
            })

            return {
                "trace": self.trace,
                "final_output": output,
                "session_id": self.session_id
            }

        except Exception as e:
            logger.error(f"Error in route: {str(e)}", exc_info=True)
            return {
                "trace": [{"stage": "error", "error": str(e)}],
                "final_output": "I encountered an error processing your request. Please try again.",
                "session_id": self.session_id
            }