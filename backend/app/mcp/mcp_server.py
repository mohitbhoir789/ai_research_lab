"""
MCPServer Module
Main orchestrator for the AI Research Assistant application.
Handles routing between different specialized agents.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
import asyncio

from agents.agent_core import AgentManager, BaseAgent, LLMAgent, Memory, Tool
from app.agents.retriever_agent import RetrieverAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.experimental_agent import ExperimentalAgent
from app.agents.researcher_agent import ResearcherAgent
from app.agents.critic_agent import CriticAgent
from app.agents.verifier_agent import VerifierAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.utils.llm import LLMHandler
from app.utils.memory_manager import MemoryManager
from app.utils.guardrails import GuardRailsChecker

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
        self.guardrails = GuardRailsChecker()

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
        # Initialize specialized agents with the current model and provider
        self.retriever_agent = RetrieverAgent(model=self.model, provider=self.provider, guardrails=self.guardrails)
        self.planner_agent = PlannerAgent(model=self.model, provider=self.provider, guardrails=self.guardrails)
        self.experimental_agent = ExperimentalAgent(model=self.model, provider=self.provider, guardrails=self.guardrails)
        self.researcher_agent = ResearcherAgent(model=self.model, provider=self.provider, guardrails=self.guardrails)
        self.critic_agent = CriticAgent(model=self.model, provider=self.provider, guardrails=self.guardrails)
        self.verifier_agent = VerifierAgent(model=self.model, provider=self.provider, guardrails=self.guardrails)
        self.summarizer_agent = SummarizerAgent(model=self.model, provider=self.provider, guardrails=self.guardrails)

        # Register all agents with the agent manager
        self.agent_manager.register_agent(self.retriever_agent)
        self.agent_manager.register_agent(self.planner_agent)
        self.agent_manager.register_agent(self.experimental_agent)
        self.agent_manager.register_agent(self.researcher_agent)
        self.agent_manager.register_agent(self.critic_agent)
        self.agent_manager.register_agent(self.verifier_agent)
        self.agent_manager.register_agent(self.summarizer_agent)

        # Set the researcher agent as the default active agent
        self.agent_manager.set_active_agent(self.researcher_agent.id)

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
                "final_output": guardrail_check["message"]
            }

        # Simple conversational replies
        greetings = ["hi", "hello", "hey"]
        farewells = ["bye", "goodbye"]
        thanks = ["thanks", "thank you"]

        lower_input = user_input.lower().strip()
        if lower_input in greetings:
            return {"trace": [{"stage": "conversation", "result": "greeting"}], "final_output": "Hello! How can I assist you with your research today?"}
        elif lower_input in farewells:
            return {"trace": [{"stage": "conversation", "result": "farewell"}], "final_output": "Goodbye! Feel free to come back if you have more questions."}
        elif lower_input in thanks:
            return {"trace": [{"stage": "conversation", "result": "gratitude"}], "final_output": "You're welcome! 😊"}

        # Detect user intent to determine appropriate workflow
        intent = await self.detect_user_intent_llm(user_input)

        # Route to appropriate workflow based on intent
        if intent == "research":
            output = await self.research_workflow(user_input)
        elif intent == "summary":
            output = await self.summary_workflow(user_input)
        elif intent == "direct_query":
            output = await self.direct_query_workflow(user_input)
        elif intent == "experimental":
            output = await self.experimental_workflow(user_input)
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

    
    async def detect_user_intent_llm(self, user_input: str) -> str:
        """
        Use LLM to detect the user's intent for routing
        
        Args:
            user_input: The user's query
            
        Returns:
            Intent as string: "research", "summary", "direct_query", "experimental"
        """
        prompt = f"""
Classify the user query into one of these categories:

1. research: If asking for research ideas, experiments, hypothesis, or in-depth research proposals
2. summary: If asking for summarization, explanation, or understanding of concepts or papers
3. direct_query: If asking a simple factual question that needs a direct answer
4. experimental: If asking for experimental design, methodology or testing procedures

Query: "{user_input}"

Respond with exactly one word from the list: research, summary, direct_query, or experimental.
"""
        response = await self.llm.generate(
            prompt=prompt,
            model=self.model,
            provider=self.provider,
            temperature=0.3,
            max_tokens=50
        )
        
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
    
    async def research_workflow(self, user_input: str) -> str:
        """
        Full research workflow using multiple specialized agents
        
        Args:
            user_input: The user's research query
            
        Returns:
            Final research output as a string
        """
        # Step 1: Retrieve background information
        retrieved_context = await self.retriever_agent.run(user_input)
        self.trace.append({
            "agent": "RetrieverAgent",
            "action": "Retrieved knowledge",
            "output": retrieved_context
        })
        
        # Step 2: Create research plan
        plan = await self.planner_agent.run(user_input)
        self.trace.append({
            "agent": "PlannerAgent",
            "action": "Created research plan",
            "output": plan
        })
        
        # Step 3: Generate experimental ideas
        experiment_prompt = f"""
Query: {user_input}

Background knowledge:
{retrieved_context}

Research plan:
{plan}

Based on this information, suggest hypotheses and experiments.
"""
        experiments = await self.experimental_agent.run(experiment_prompt)
        self.trace.append({
            "agent": "ExperimentalAgent",
            "action": "Generated experimental design",
            "output": experiments
        })
        
        # Step 4: Create full research proposal
        proposal_prompt = f"""
Query: {user_input}

Background knowledge:
{retrieved_context}

Research plan:
{plan}

Experimental ideas:
{experiments}

Based on all this information, create a complete research proposal.
"""
        research_proposal = await self.researcher_agent.run(proposal_prompt)
        self.trace.append({
            "agent": "ResearcherAgent",
            "action": "Created full research proposal",
            "output": research_proposal
        })
        
        # Step 5: Critical review of the proposal
        critique_prompt = f"""
Research proposal to review:

{research_proposal}

Provide a critical analysis of this research proposal.
"""
        critique = await self.critic_agent.run(critique_prompt)
        self.trace.append({
            "agent": "CriticAgent",
            "action": "Provided critical review",
            "output": critique
        })
        
        # Step 6: Verify facts and feasibility
        verification_prompt = f"""
Research proposal:

{research_proposal}

Critique:
{critique}

Verify the facts, data sources, and overall feasibility of this research proposal.
"""
        verification = await self.verifier_agent.run(verification_prompt)
        self.trace.append({
            "agent": "VerifierAgent",
            "action": "Verified facts and feasibility",
            "output": verification
        })
        
        # Step 7: Final academic summary
        final_prompt = f"""
# 🧠 Research Proposal

{research_proposal}

---

# 📝 Critique and Challenges

{critique}

---

# 🔎 Verification

{verification}

---

Create a comprehensive final academic report that incorporates all the above components.
Format the output with clear sections and subsections using markdown.
"""
        final_summary = await self.summarizer_agent.run(final_prompt)
        self.trace.append({
            "agent": "SummarizerAgent",
            "action": "Created final summary",
            "output": final_summary
        })
        
        # Return the final summary
        return final_summary
    
    async def summary_workflow(self, user_input: str) -> str:
        """
        Workflow for summarizing content
        
        Args:
            user_input: The user's query for summarization
            
        Returns:
            Summarized content as a string
        """
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
    
    async def direct_query_workflow(self, user_input: str) -> str:
        """
        Workflow for answering direct factual queries
        
        Args:
            user_input: The user's direct question
            
        Returns:
            Answer as a string
        """
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
    
    async def experimental_workflow(self, user_input: str) -> str:
        """
        Workflow for designing experiments
        
        Args:
            user_input: The user's experimental design query
            
        Returns:
            Experimental design as a string
        """
        # Step 1: Retrieve background information
        retrieved_context = await self.retriever_agent.run(user_input)
        self.trace.append({
            "agent": "RetrieverAgent",
            "action": "Retrieved knowledge",
            "output": retrieved_context
        })
        
        # Step 2: Generate experimental design
        experiment_prompt = f"""
Query: {user_input}

Background knowledge:
{retrieved_context}

Please design a detailed experimental protocol including:
1. Hypothesis
2. Materials and methods
3. Experimental setup
4. Data collection procedures
5. Analysis methods
6. Expected outcomes and interpretations

Format the output with clear sections using markdown.
"""
        experiments = await self.experimental_agent.run(experiment_prompt)
        self.trace.append({
            "agent": "ExperimentalAgent",
            "action": "Generated experimental design",
            "output": experiments
        })
        
        # Step 3: Critical review of the experimental design
        critique_prompt = f"""
Experimental design to review:

{experiments}

Provide a critical analysis of this experimental design, focusing on:
1. Scientific rigor
2. Potential confounding variables
3. Statistical power
4. Feasibility
5. Ethical considerations
"""
        critique = await self.critic_agent.run(critique_prompt)
        self.trace.append({
            "agent": "CriticAgent",
            "action": "Provided critical review",
            "output": critique
        })
        
        # Step 4: Final revised experimental design
        final_prompt = f"""
# 🧪 Original Experimental Design

{experiments}

---

# 📝 Critique and Challenges

{critique}

---

Based on the original design and critique, create a revised experimental protocol
that addresses the identified issues. Format the output with clear sections using markdown.
"""
        final_design = await self.experimental_agent.run(final_prompt)
        self.trace.append({
            "agent": "ExperimentalAgent",
            "action": "Created final experimental design",
            "output": final_design
        })
        
        return final_design
    
    async def handle_file_upload(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Process an uploaded file and extract relevant information
        
        Args:
            file_path: Path to the uploaded file
            file_type: Type of the file (pdf, csv, txt, etc.)
            
        Returns:
            Dict containing extraction results
        """
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
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions
        
        Returns:
            List of session information dictionaries
        """
        return self.memory_manager.list_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its associated data
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            Boolean indicating success
        """
        # Clear current session if it's the one being deleted
        if session_id == self.session_id:
            self.session_id = None
            self.global_context = {}
        
        return self.memory_manager.delete_session(session_id)
    
    def get_agents_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered agents
        
        Returns:
            List of agent information dictionaries
        """
        return self.agent_manager.list_agents()
    
    def set_active_agent(self, agent_id: str) -> bool:
        """
        Set an agent as the active agent
        
        Args:
            agent_id: ID of the agent to set as active
            
        Returns:
            Boolean indicating success
        """
        return self.agent_manager.set_active_agent(agent_id)
    
    async def generate_session_summary(self, session_id: Optional[str] = None) -> str:
        """
        Generate a summary of the current or specified session
        
        Args:
            session_id: Optional session ID to summarize
            
        Returns:
            Summary as a string
        """
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


# For testing the MCP server directly
if __name__ == "__main__":
    import asyncio
    
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