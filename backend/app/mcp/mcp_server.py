# backend/app/mcp/mcp_server.py

from app.agents.retriever_agent import RetrieverAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.experimental_agent import ExperimentalAgent
from app.agents.researcher_agent import ResearcherAgent
from app.agents.critic_agent import CriticAgent
from app.agents.verifier_agent import VerifierAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.utils.llm import LLMHandler
import json

class MCPServer:
    def __init__(self):
        self.retriever_agent = RetrieverAgent()
        self.planner_agent = PlannerAgent()
        self.experimental_agent = ExperimentalAgent()
        self.researcher_agent = ResearcherAgent()
        self.critic_agent = CriticAgent()
        self.verifier_agent = VerifierAgent()
        self.summarizer_agent = SummarizerAgent()
        self.llm = LLMHandler()
        self.trace = []  # For step-by-step debugging

    async def route(self, user_input: str):
        """Main router deciding research or summary."""
        self.trace = []  # Reset trace
        intent = await self.detect_user_intent_llm(user_input)

        if intent == "research":
            output = await self.research_workflow(user_input)
        else:
            output = await self.summary_workflow(user_input)

        # 🚀 Print beautiful JSON Trace
        print("\n===== TRACE JSON START =====\n")
        print(json.dumps(self.trace, indent=2))
        print("\n===== TRACE JSON END =====\n")

        return {
            "trace": self.trace,
            "final_output": output
        }

    async def detect_user_intent_llm(self, user_input: str) -> str:
        """Use LLM to detect if it's research or basic summary."""
        prompt = f"""
Classify the user query.

- If asking for research ideas, experiments, hypothesis: reply only research
- If asking for summary, explanation, understanding: reply only summary

Query: "{user_input}"

Respond exactly with one word: research OR summary.
"""
        response = await self.llm.generate(prompt, model="groq")
        return "research" if "research" in response.lower() else "summary"

    async def research_workflow(self, user_input: str) -> str:
        """Full research plan building workflow."""

        # Step 1: Retrieve background
        retrieved_context = await self.retriever_agent.run(user_input)
        self.trace.append({
            "agent": "RetrieverAgent",
            "action": "Retrieved knowledge (books, papers, wiki)",
            "output": retrieved_context
        })

        # Step 2: Plan the research
        plan = await self.planner_agent.run(user_input)
        self.trace.append({
            "agent": "PlannerAgent",
            "action": "Outlined research plan",
            "output": plan
        })

        # Step 3: Generate experimental ideas
        experiments = await self.experimental_agent.run(user_input)
        self.trace.append({
            "agent": "ExperimentalAgent",
            "action": "Suggested hypotheses and experiments",
            "output": experiments
        })

        # Step 4: Full research proposal
        research_proposal = await self.researcher_agent.run(user_input)
        self.trace.append({
            "agent": "ResearcherAgent",
            "action": "Generated detailed research proposal",
            "output": research_proposal
        })

        # Step 5: Critique the proposal
        critique = await self.critic_agent.run(user_input)
        self.trace.append({
            "agent": "CriticAgent",
            "action": "Peer-reviewed the proposal",
            "output": critique
        })

        # Step 6: Fact verification
        verification = await self.verifier_agent.run(user_input)
        self.trace.append({
            "agent": "VerifierAgent",
            "action": "Verified facts and feasibility",
            "output": verification
        })

        # Step 7: Final academic summary
        combined_input = f"""
# 🧠 Research Proposal

{research_proposal}

---

# 📝 Critique and Challenges

{critique}

---

# 🔎 Verification

{verification}
"""
        final_summary = await self.summarizer_agent.run(combined_input)
        self.trace.append({
            "agent": "SummarizerAgent",
            "action": "Produced final academic report",
            "output": final_summary
        })

        return final_summary

    async def summary_workflow(self, user_input: str) -> str:
        """Simple explanation/summary for easy queries."""
        
        # Step 1: Retrieve simple context
        retrieved_context = await self.retriever_agent.run(user_input)
        self.trace.append({
            "agent": "RetrieverAgent",
            "action": "Fetched relevant wiki/book context",
            "output": retrieved_context
        })

        # Step 2: Generate short summary
        final_summary = await self.summarizer_agent.run(user_input)
        self.trace.append({
            "agent": "SummarizerAgent",
            "action": "Generated short explanation",
            "output": final_summary
        })

        return final_summary

# ✅ End of MCPServer