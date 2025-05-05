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
            # Start or resume session
            if session_id != self.session_id:
                await self.start_session(session_id)

            # Reset trace for new request
            self.trace = []

            # Handle simple conversational replies
            lower_input = user_input.lower().strip()
            if lower_input in ["hi", "hello", "hey"]:
                return {
                    "trace": [{"stage": "conversation", "result": "greeting"}], 
                    "final_output": "Hello! How can I assist you today?",
                    "session_id": self.session_id
                }
            elif lower_input in ["bye", "goodbye"]:
                return {
                    "trace": [{"stage": "conversation", "result": "farewell"}], 
                    "final_output": "Goodbye! Feel free to come back if you have more questions.",
                    "session_id": self.session_id
                }
            elif lower_input in ["thanks", "thank you"]:
                return {
                    "trace": [{"stage": "conversation", "result": "gratitude"}], 
                    "final_output": "You're welcome! 😊",
                    "session_id": self.session_id
                }

            # Detect intent and validate against guardrails using the IntentDetectorAgent
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

            # Add user input to global context
            if "conversation_history" not in self.global_context:
                self.global_context["conversation_history"] = []

            self.global_context["conversation_history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": self.memory_manager.get_timestamp(),
                "mode": mode,
                "intent": intent_data
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

            # Add response to global context
            self.global_context["conversation_history"].append({
                "role": "assistant",
                "content": output,
                "timestamp": self.memory_manager.get_timestamp(),
                "mode": mode
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

    async def _run_with_fallback(self, func, *args, **kwargs) -> Optional[Any]:
        """
        Run a function with provider fallback if it fails
        
        Args:
            func: The function to run
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The function result or None if all attempts fail
        """
        # Try with original provider first
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Function {func.__name__} failed with provider {self.provider}: {str(e)}")
            
            # Try with each fallback provider
            original_provider = self.provider
            original_model = self.model
            
            for fallback_attempt in range(2):  # Try two fallback attempts
                try:
                    # Get next fallback provider
                    fallback_provider, fallback_model = self.get_fallback_provider(self.provider)
                    
                    # Log fallback attempt
                    logger.info(f"Trying fallback to {fallback_provider}/{fallback_model}")
                    
                    # Update model settings temporarily 
                    self.update_model_settings(fallback_model, fallback_provider)
                    
                    # Try function with fallback provider
                    result = await func(*args, **kwargs)
                    
                    # If successful, log it
                    logger.info(f"Fallback to {fallback_provider}/{fallback_model} succeeded")
                    
                    # Add to trace
                    self.trace.append({
                        "fallback": True,
                        "original_provider": original_provider,
                        "new_provider": fallback_provider,
                        "function": func.__name__
                    })
                    
                    return result
                except Exception as fallback_e:
                    logger.warning(f"Fallback {fallback_attempt+1} to {self.provider} failed: {str(fallback_e)}")
            
            # If all fallbacks fail, restore original settings
            self.update_model_settings(original_model, original_provider)
            
            # Add failure to trace
            self.trace.append({
                "fallback_failed": True,
                "function": func.__name__,
                "error": str(e)
            })
            
            return None

    async def direct_query_workflow(self, user_input: str) -> str:
        """
        Handle direct query workflow for brief answers in chat mode

        Args:
            user_input: The user's query

        Returns:
            Brief response to the query
        """
        try:
            # Use summarizer agent with specific instructions for brief answer
            config = {
                "style": "brief",
                "max_length": 150  # Keep it concise
            }
            
            # Handoff to summarizer agent with reasoning
            handoff_result = await self.agent_handoff(
                from_agent_id=self.researcher_agent.agent_id,  # Default agent
                to_agent_id=self.summarizer_agent.agent_id,
                reason="Query requires a brief, direct answer with minimal detail",
                user_input=user_input,
                context=config
            )
            
            if "error" in handoff_result:
                logger.error(f"Agent handoff failed: {handoff_result['error']}")
                # Fallback to direct call if handoff fails
                return await self.summarizer_agent.run(user_input)
                
            return handoff_result["response"]
            
        except Exception as e:
            logger.error(f"Error in direct_query_workflow: {str(e)}")
            raise  # Let the _run_with_fallback function handle it

    async def summary_workflow(self, user_input: str) -> str:
        """
        Handle summary workflow for detailed explanations in chat mode

        Args:
            user_input: The user's query

        Returns:
            Detailed explanation response to the query
        """
        try:
            # First handoff to retriever agent to get relevant context
            retrieval_result = await self.agent_handoff(
                from_agent_id=self.researcher_agent.agent_id,  # Default agent
                to_agent_id=self.retriever_agent.agent_id,
                reason="Query requires relevant context retrieval before generating a detailed explanation",
                user_input=user_input,
                context={"task": "retrieval", "mode": "detailed"}
            )
            
            if "error" in retrieval_result:
                logger.error(f"Retriever agent handoff failed: {retrieval_result['error']}")
                # Fallback to direct call if handoff fails
                context = await self.retriever_agent.run(user_input)
            else:
                context = retrieval_result["response"]
            
            # Now handoff to summarizer agent with the retrieved context
            summarizer_result = await self.agent_handoff(
                from_agent_id=self.retriever_agent.agent_id,
                to_agent_id=self.summarizer_agent.agent_id,
                reason="Context retrieved, now generate a comprehensive explanation based on the gathered information",
                user_input=user_input,
                context={
                    "retrieved_context": context,
                    "style": "detailed",
                    "task": "explanation"
                }
            )
            
            if "error" in summarizer_result:
                logger.error(f"Summarizer agent handoff failed: {summarizer_result['error']}")
                # Fallback to direct call with enhanced prompt if handoff fails
                enhanced_prompt = f"""
Query: {user_input}

Based on the following context, provide a detailed explanation:

{context}
"""
                return await self.summarizer_agent.run(enhanced_prompt)
            
            return summarizer_result["response"]
            
        except Exception as e:
            logger.error(f"Error in summary_workflow: {str(e)}")
            raise  # Let the _run_with_fallback function handle it

    async def research_workflow(self, user_input: str) -> str:
        """
        Handle research workflow with paper retrieval for research mode

        Args:
            user_input: The user's research query

        Returns:
            Research-oriented response with paper references
        """
        try:
            # First handoff to retriever agent to get research papers and context
            retrieval_result = await self.agent_handoff(
                from_agent_id=self.researcher_agent.agent_id,  # Default agent
                to_agent_id=self.retriever_agent.agent_id,
                reason="Research query requires specialized papers and context retrieval for in-depth analysis",
                user_input=user_input,
                context={"task": "research_retrieval", "mode": "comprehensive", "require_papers": True}
            )
            
            if "error" in retrieval_result:
                logger.error(f"Retriever agent handoff failed: {retrieval_result['error']}")
                # Fallback to direct call if handoff fails
                context = await self.retriever_agent.run(user_input)
            else:
                context = retrieval_result["response"]
            
            # Now handoff to researcher agent with the retrieved context
            research_result = await self.agent_handoff(
                from_agent_id=self.retriever_agent.agent_id,
                to_agent_id=self.researcher_agent.agent_id,
                reason="Relevant papers retrieved, now generating in-depth research analysis based on academic sources",
                user_input=user_input,
                context={
                    "retrieved_papers": context,
                    "task": "research_analysis",
                    "depth": "academic"
                }
            )
            
            if "error" in research_result:
                logger.error(f"Researcher agent handoff failed: {research_result['error']}")
                # Fallback to direct call with enhanced prompt if handoff fails
                enhanced_prompt = f"""
Research Query: {user_input}

Based on the following research context and papers, provide an in-depth analysis:

{context}

Include relevant research findings, methodologies, and directions for further exploration.
"""
                response = await self.researcher_agent.run(enhanced_prompt)
            else:
                response = research_result["response"]
            
            # Handoff to verifier agent to verify the research response
            verification_result = await self.agent_handoff(
                from_agent_id=self.researcher_agent.agent_id,
                to_agent_id=self.verifier_agent.agent_id,
                reason="Research analysis completed, now verifying academic accuracy and removing any potential errors",
                user_input=user_input,
                context={
                    "research_response": response,
                    "task": "verification",
                    "standard": "academic_rigor"
                }
            )
            
            if "error" in verification_result:
                logger.error(f"Verifier agent handoff failed: {verification_result['error']}")
                return response  # Return unverified response if verification fails
                
            verified_response = verification_result["response"]
            
            # Return either the verified response or the original if verification isn't appropriate
            if "Error" in verified_response or len(verified_response) < len(response) / 2:
                return response
            
            return verified_response
            
        except Exception as e:
            logger.error(f"Error in research_workflow: {str(e)}")
            raise  # Let the _run_with_fallback function handle it

    async def agent_handoff(self, from_agent_id: str, to_agent_id: str, reason: str, user_input: str, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handoff control from one agent to another with explicit reasoning

        Args:
            from_agent_id: ID of the agent handing off control
            to_agent_id: ID of the agent receiving control
            reason: Reasoning for why this handoff is occurring
            user_input: The original user query or modified query
            context: Optional context to pass to the receiving agent

        Returns:
            Dict containing the receiving agent's response and metadata
        """
        try:
            # Get the agent instances
            from_agent = self.agent_manager.get_agent_by_id(from_agent_id)
            to_agent = self.agent_manager.get_agent_by_id(to_agent_id)
            
            if not from_agent or not to_agent:
                logger.error(f"Handoff failed: Invalid agent IDs - from: {from_agent_id}, to: {to_agent_id}")
                return {"error": "Invalid agent IDs for handoff"}
            
            # Record handoff in trace
            handoff_record = {
                "timestamp": self.memory_manager.get_timestamp(),
                "from_agent": from_agent.__class__.__name__,
                "to_agent": to_agent.__class__.__name__,
                "reason": reason,
                "query": user_input[:100] + ("..." if len(user_input) > 100 else "")  # Truncate for logging
            }
            
            self.trace.append({
                "stage": "agent_handoff",
                "from": from_agent.__class__.__name__,
                "to": to_agent.__class__.__name__, 
                "reason": reason
            })
            
            # Store handoff in history
            if not hasattr(self, 'handoff_history'):
                self.handoff_history = []
            self.handoff_history.append(handoff_record)
            
            # Set the active agent in the agent manager
            self.agent_manager.set_active_agent(to_agent_id)
            
            # Prepare enhanced prompt with reasoning and context
            enhanced_input = user_input
            if context:
                context_str = "\n\n".join([f"{k}: {v}" for k, v in context.items()])
                enhanced_input = f"""
Query: {user_input}

Context:
{context_str}

Reasoning: {reason}
"""
            
            # Execute the receiving agent
            logger.info(f"Agent handoff: {from_agent.__class__.__name__} -> {to_agent.__class__.__name__}, Reason: {reason}")
            response = await to_agent.run(enhanced_input)
            
            return {
                "agent": to_agent.__class__.__name__,
                "response": response,
                "handoff_record": handoff_record
            }
            
        except Exception as e:
            logger.error(f"Error in agent_handoff: {str(e)}")
            return {"error": f"Handoff failed: {str(e)}"}