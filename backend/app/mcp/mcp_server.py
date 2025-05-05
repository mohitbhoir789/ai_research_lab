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
from typing import Dict, Any, List, Optional
import asyncio

# Fix import path consistency
from app.agents.agent_core import AgentManager, BaseAgent, LLMAgent, Memory, Tool
from app.agents.retriever_agent import RetrieverAgent
#
from app.agents.researcher_agent import ResearcherAgent
#
from app.agents.verifier_agent import VerifierAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.utils.llm import LLMHandler
from app.utils.memory_manager import MemoryManager
from app.utils.guardrails import GuardrailsChecker

logger = logging.getLogger(__name__)

class MCPServer:
    """
    Master Control Program Server - Orchestrates all agent interactions
    """

    def __init__(self, model="llama3-70b-8192", provider="groq"):
        """Initialize the MCP server with all specialized agents"""
        # Initialize LLM handler for utility functions
        self.llm = LLMHandler()

        # Initialize agent manager
        self.agent_manager = AgentManager()

        # Initialize memory manager for persistent memory across sessions
        self.memory_manager = MemoryManager()

        # Initialize guardrails checker
        self.guardrails = GuardrailsChecker()

        # Default model and provider
        self.model = model
        self.provider = provider

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
            self.retriever_agent = RetrieverAgent(model=self.model, provider=self.provider)
            self.researcher_agent = ResearcherAgent(model=self.model, provider=self.provider)
            self.verifier_agent = VerifierAgent(model=self.model, provider=self.provider)
            self.summarizer_agent = SummarizerAgent(model=self.model, provider=self.provider)

            # Register all agents with the agent manager
            self.agent_manager.register_agent(self.retriever_agent)
            self.agent_manager.register_agent(self.researcher_agent)
            self.agent_manager.register_agent(self.verifier_agent)
            self.agent_manager.register_agent(self.summarizer_agent)

            # Set the researcher agent as the default active agent
            self.agent_manager.set_active_agent(self.researcher_agent.agent_id)
            
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
        self.model = model
        self.provider = provider

        # Re-initialize all agents with new settings
        self._initialize_agents()

        logger.info(f"Updated model settings: model={model}, provider={provider}")

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

    async def route(self, user_input: str, session_id: Optional[str] = None, uploaded_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main entry point for routing user requests to appropriate workflows

        Args:
            user_input: The user's query or request
            session_id: Optional session ID to maintain context
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

            # Add user input to global context
            if "conversation_history" not in self.global_context:
                self.global_context["conversation_history"] = []

            self.global_context["conversation_history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": self.memory_manager.get_timestamp()
            })

            # Add uploaded files to context if provided
            if uploaded_files:
                self.global_context["uploaded_files"] = uploaded_files
                logger.info(f"Added uploaded files to context: {uploaded_files}")

            # Check input guardrails
            guardrail_check = self.guardrails.check_input(user_input)
            if not guardrail_check["passed"]:
                return {
                    "trace": [{
                        "stage": "guardrails",
                        "result": "rejected",
                        "reason": guardrail_check["reason"]
                    }],
                    "final_output": guardrail_check["message"],
                    "session_id": self.session_id
                }

            # Simple conversational replies
            greetings = ["hi", "hello", "hey"]
            farewells = ["bye", "goodbye"]
            thanks = ["thanks", "thank you"]

            lower_input = user_input.lower().strip()
            if lower_input in greetings:
                return {"trace": [{"stage": "conversation", "result": "greeting"}], 
                        "final_output": "Hello! How can I assist you with your research today?",
                        "session_id": self.session_id}
            elif lower_input in farewells:
                return {"trace": [{"stage": "conversation", "result": "farewell"}], 
                        "final_output": "Goodbye! Feel free to come back if you have more questions.",
                        "session_id": self.session_id}
            elif lower_input in thanks:
                return {"trace": [{"stage": "conversation", "result": "gratitude"}], 
                        "final_output": "You're welcome! 😊",
                        "session_id": self.session_id}

            # Detect user intent to determine appropriate workflow
            intent = await self.detect_user_intent_llm(user_input)

            # Route to appropriate workflow based on intent
            if intent == "research":
                output = await self.research_workflow(user_input)
            elif intent == "summary":
                output = await self.summary_workflow(user_input)
            elif intent == "direct_query":
                output = await self.direct_query_workflow(user_input)
            else:
                # Default to researcher agent for unknown intents
                output = await self.researcher_agent.run(user_input)

            # Validate and sanitize output
            output = self.guardrails.sanitize_output(output)

            # Add assistant response to global context
            self.global_context["conversation_history"].append({
                "role": "assistant",
                "content": output,
                "timestamp": self.memory_manager.get_timestamp()
            })

            # Save session state
            self.memory_manager.save_session(self.session_id, {
                "global_context": self.global_context,
                "last_trace": self.trace
            })

            # Log the trace for debugging
            logger.debug(f"Execution trace: {json.dumps(self.trace, indent=2)}")

            # Return both trace and final output
            return {
                "trace": self.trace,
                "final_output": output,
                "session_id": self.session_id
            }
        except Exception as e:
            logger.error(f"Error in route: {str(e)}", exc_info=True)
            # Provide a fallback response in case of error
            return {
                "trace": [{"stage": "error", "error": str(e)}],
                "final_output": "I encountered an error processing your request. Please try again.",
                "session_id": self.session_id
            }

    
    async def detect_user_intent_llm(self, user_input: str) -> str:
        """
        Use LLM to detect the user's intent for routing
        
        Args:
            user_input: The user's query
            
        Returns:
            Intent as string: "research", "summary", "direct_query", "experimental"
        """
        try:
            prompt = f"""
Classify the user query into one of these categories:

1. research: If asking for research ideas, experiments, hypothesis, or in-depth research proposals
2. summary: If asking for summarization, explanation, or understanding of concepts or papers
3. direct_query: If asking a simple factual question that needs a direct answer
4. experimental: If asking for experimental design, methodology or testing procedures

Query: "{user_input}"

Respond with exactly one word from the list: research, summary, direct_query, or experimental.
"""
            from backend.app.utils.llm import LLMConfig, LLMProvider
            config = LLMConfig(
                model=self.model,
                provider=LLMProvider(self.provider),
                temperature=0.3,
                max_tokens=50
            )
            
            response, _ = await self.llm.generate(prompt=prompt, config=config)
            
            # Normalize the response
            normalized_response = response.lower().strip()
            
            # Map to valid intents
            if "research" in normalized_response:
                intent = "research"
            elif "summary" in normalized_response:
                intent = "summary"
            elif "direct" in normalized_response or "query" in normalized_response:
                intent = "direct_query"
            elif "experiment" in normalized_response:
                intent = "experimental"
            else:
                # Default to research for unclear responses
                intent = "research"
            
            # Add detection to trace
            self.trace.append({
                "stage": "intent_detection",
                "result": intent,
                "input": user_input
            })
            
            return intent
        except Exception as e:
            logger.error(f"Error in intent detection: {str(e)}")
            # Default to research in case of error
            self.trace.append({
                "stage": "intent_detection",
                "result": "research (default due to error)",
                "error": str(e)
            })
            return "research"
    
    async def research_workflow(self, user_input: str) -> str:
        """
        Full research workflow using multiple specialized agents
        
        Args:
            user_input: The user's research query
            
        Returns:
            Final research output as a string
        """
        try:
            # Step 1: Retrieve background information
            retrieved_context = await self.retriever_agent.run(user_input)
            self.trace.append({
                "agent": "RetrieverAgent",
                "action": "Retrieved knowledge",
                "output": retrieved_context
            })

            # Step 2: Create full research proposal (plan/experiments handled in prompt)
            proposal_prompt = f"""
Query: {user_input}

Background knowledge:
{retrieved_context}

Use the background information above to create a comprehensive research proposal.
"""
            research_proposal = await self.researcher_agent.run(proposal_prompt)
            self.trace.append({
                "agent": "ResearcherAgent",
                "action": "Created full research proposal",
                "output": research_proposal
            })

            # Step 3: Verify facts and feasibility
            verification_prompt = f"""
Research proposal:

{research_proposal}

Please verify the facts, check feasibility, and briefly mention any risks or gaps.
"""
            verification = await self.verifier_agent.run(verification_prompt)
            self.trace.append({
                "agent": "VerifierAgent",
                "action": "Verified facts and feasibility",
                "output": verification
            })

            # Step 4: Final academic summary
            final_prompt = f"""
# 🧠 Research Proposal

{research_proposal}

---

# 🔎 Verification

{verification}

---

Create a final academic report that integrates the proposal and verification.
Use markdown with clear sections.
"""
            final_summary = await self.summarizer_agent.run(final_prompt)
            self.trace.append({
                "agent": "SummarizerAgent",
                "action": "Created final summary",
                "output": final_summary
            })

            # Return the final summary
            return final_summary

        except Exception as e:
            logger.error(f"Error in research workflow: {str(e)}")
            return f"I encountered an error while researching this topic. Please try a different query or try again later."
    
    async def summary_workflow(self, user_input: str) -> str:
        """
        Workflow for summarizing content
        
        Args:
            user_input: The user's query for summarization
            
        Returns:
            Summarized content as a string
        """
        try:
            # Step 1: Retrieve relevant information
            retrieved_context = await self.retriever_agent.run(user_input)
            self.trace.append({
                "agent": "RetrieverAgent",
                "action": "Retrieved knowledge",
                "output": retrieved_context
            })
            
            # Step 2: Generate summary
            summary_prompt = f"""
Query: {user_input}

Background knowledge:
{retrieved_context}

Please provide a clear, concise summary based on this information.
Use markdown formatting for readability.
"""
            summary = await self.summarizer_agent.run(summary_prompt)
            self.trace.append({
                "agent": "SummarizerAgent",
                "action": "Generated summary",
                "output": summary
            })
            
            # Step 3: Verify facts in the summary
            verification_prompt = f"""
Summary to verify:

{summary}

Please verify the factual accuracy of this summary and correct any inaccuracies.
"""
            verified_summary = await self.verifier_agent.run(verification_prompt)
            self.trace.append({
                "agent": "VerifierAgent",
                "action": "Verified summary",
                "output": verified_summary
            })
            
            # Return the verified summary
            return verified_summary
            
        except Exception as e:
            logger.error(f"Error in summary workflow: {str(e)}")
            return f"I encountered an error while generating a summary. Please try a different query or try again later."
    
    async def direct_query_workflow(self, user_input: str) -> str:
        """
        Workflow for answering direct factual queries
        
        Args:
            user_input: The user's direct question
            
        Returns:
            Answer as a string
        """
        try:
            # Use the researcher agent directly for simple factual queries
            response = await self.researcher_agent.run(user_input)
            self.trace.append({
                "agent": "ResearcherAgent",
                "action": "Answered direct query",
                "output": response
            })
            
            # Verify the response
            verification_prompt = f"""
Query: {user_input}

Response:
{response}

Please verify the factual accuracy of this response and provide a corrected
version if necessary. Be concise but thorough.
"""
            verified_response = await self.verifier_agent.run(verification_prompt)
            self.trace.append({
                "agent": "VerifierAgent",
                "action": "Verified response",
                "output": verified_response
            })
            
            return verified_response
            
        except Exception as e:
            logger.error(f"Error in direct query workflow: {str(e)}")
            return f"I encountered an error while answering your question. Please try rephrasing or ask a different question."
    
    
    async def handle_file_upload(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Process an uploaded file and extract relevant information
        
        Args:
            file_path: Path to the uploaded file
            file_type: Type of the file (pdf, csv, txt, etc.)
            
        Returns:
            Dict containing extraction results
        """
        try:
            # Add file info to global context
            if "uploaded_files" not in self.global_context:
                self.global_context["uploaded_files"] = []
            
            self.global_context["uploaded_files"].append({
                "path": file_path,
                "type": file_type,
                "timestamp": self.memory_manager.get_timestamp()
            })
            
            # Process file based on type
            # This would integrate with tools that can read PDFs, CSVs, etc.
            extraction_prompt = f"""
Analyze the uploaded {file_type} file and extract key information.
Provide a concise summary of the file contents.
"""
            
            # Use the retriever agent to process the file
            # In a real implementation, this would involve a tool that can read the file
            summary = await self.retriever_agent.run(extraction_prompt)
            
            result = {
                "file_path": file_path,
                "file_type": file_type,
                "summary": summary
            }
            
            # Add to trace
            self.trace.append({
                "stage": "file_processing",
                "file_type": file_type,
                "result": "processed"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {
                "file_path": file_path,
                "file_type": file_type,
                "error": str(e),
                "summary": "Failed to process file"
            }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions
        
        Returns:
            List of session information dictionaries
        """
        try:
            return self.memory_manager.list_sessions()
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its associated data
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            # Clear current session if it's the one being deleted
            if session_id == self.session_id:
                self.session_id = None
                self.global_context = {}
            
            return self.memory_manager.delete_session(session_id)
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return False
    
    def get_agents_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered agents
        
        Returns:
            List of agent information dictionaries
        """
        try:
            return self.agent_manager.list_agents()
        except Exception as e:
            logger.error(f"Error getting agent info: {str(e)}")
            return []
    
    def set_active_agent(self, agent_id: str) -> bool:
        """
        Set an agent as the active agent
        
        Args:
            agent_id: ID of the agent to set as active
            
        Returns:
            Boolean indicating success
        """
        try:
            return self.agent_manager.set_active_agent(agent_id)
        except Exception as e:
            logger.error(f"Error setting active agent: {str(e)}")
            return False
    
    async def generate_session_summary(self, session_id: Optional[str] = None) -> str:
        """
        Generate a summary of the current or specified session
        
        Args:
            session_id: Optional session ID to summarize
            
        Returns:
            Summary as a string
        """
        try:
            target_session = session_id or self.session_id
            
            if not target_session:
                return "No active session to summarize."
            
            # Load session data
            session_data = self.memory_manager.load_session(target_session)
            
            if not session_data or "global_context" not in session_data:
                return "Session data not found or empty."
            
            # Extract conversation history
            history = session_data["global_context"].get("conversation_history", [])
            
            if not history:
                return "No conversation history in this session."
            
            # Format conversation for summarization
            conversation = "\n\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in history
            ])
            
            # Generate summary
            summary_prompt = f"""
Please create a concise summary of the following conversation between a user and an AI research assistant.
Focus on the main topics discussed, questions asked, and key information provided.

Conversation:
{conversation}

Summary:
"""
            
            summary = await self.summarizer_agent.run(summary_prompt)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")
            return "Failed to generate session summary due to an error."


# For testing the MCP server directly
if __name__ == "__main__":
    import asyncio
    import time  # Added for fallback session ID generation
    
    async def test_mcp():
        mcp = MCPServer()
        session_id = await mcp.start_session()
        
        print(f"Started session: {session_id}")
        
        research_query = "What are the latest advances in quantum computing and how might they impact machine learning?"
        result = await mcp.route(research_query, session_id)
        
        print("\n=== RESEARCH WORKFLOW RESULT ===")
        print(result["final_output"])
        
        summary_query = "Summarize the key concepts in transformer neural networks"
        result = await mcp.route(summary_query, session_id)
        
        print("\n=== SUMMARY WORKFLOW RESULT ===")
        print(result["final_output"])
        
        session_summary = await mcp.generate_session_summary(session_id)
        
        print("\n=== SESSION SUMMARY ===")
        print(session_summary)
    
    asyncio.run(test_mcp())