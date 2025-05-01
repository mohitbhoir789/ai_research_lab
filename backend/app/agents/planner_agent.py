"""
Planner Agent Module
Specialized agent for generating structured research plans based on user queries and goals.
"""

import logging
from app.agents.agent_core import LLMAgent
from app.utils.guardrails import GuardRailsChecker

logger = logging.getLogger(__name__)


class PlannerAgent(LLMAgent):
    """Agent specialized in constructing research plans, outlining methodologies, and breaking down complex objectives."""

    def __init__(self,
                 name: str = "Planner Agent",
                 model: str = "llama3-70b-8192",
                 temperature: float = 0.4,
                 max_tokens: int = 1500,
                 provider: str = "groq",
                 **kwargs):

        description = "I specialize in breaking down research queries into structured plans, including objectives, methods, and milestones."

        super().__init__(
            name=name,
            description=description,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            **kwargs
        )

        self.guardrails = GuardRailsChecker()

        # Specialized system instructions
        specialized_prompt = """
        As a Planner Agent, your job is to break down high-level research questions or goals into structured, logical plans.

        When creating a plan:
        1. Define the main research objectives or questions
        2. Break them down into sub-goals or hypotheses
        3. Identify required background knowledge or literature
        4. Outline possible methods or approaches to address each goal
        5. List potential tools, frameworks, or datasets needed
        6. Suggest a phased timeline or order of operations if applicable
        7. Use clear markdown formatting with bullet points and headings
        """
        self.update_system_prompt(self.system_prompt + specialized_prompt)

    async def run(self, query: str) -> str:
        """
        Entry point for generating a structured plan based on a user query.

        Args:
            query: A research question, goal, or topic

        Returns:
            A formatted plan as a markdown string
        """
        check = self.guardrails.check_input(query)
        if not check["passed"]:
            return f"⚠️ Input blocked by guardrails: {check['message']}"

        processed_prompt = self._preprocess_query(query)
        result = await self.process(processed_prompt)
        return self._postprocess_response(self.guardrails.sanitize_output(result.get("response", "")))

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess user input to guide plan generation.

        Args:
            query: Raw user query

        Returns:
            Formatted prompt
        """
        return f"""Create a structured research plan for the following query or goal:

{query}

Use markdown formatting with clear sections and bullet points. Be thorough but concise."""

    def _postprocess_response(self, response: str) -> str:
        """
        Postprocess and clean LLM output.

        Args:
            response: Raw LLM output

        Returns:
            Cleaned markdown output
        """
        if not response.strip():
            return "⚠️ No response generated."

        if not response.lower().startswith("# research plan"):
            response = "# Research Plan\n\n" + response

        return response


# For testing the agent directly
if __name__ == "__main__":
    import asyncio

    async def test_planner():
        agent = PlannerAgent()
        query = "Develop a research plan to study the impact of LLMs in clinical decision support systems."
        result = await agent.run(query)
        print(result)

    asyncio.run(test_planner())