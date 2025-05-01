"""
Researcher Agent Module
An expert agent designed to assist with academic research by generating structured proposals, hypotheses, and citations.
"""

import logging
from app.agents.agent_core import LLMAgent
from app.utils.guardrails import GuardRailsChecker

logger = logging.getLogger(__name__)


class ResearcherAgent(LLMAgent):
    """Agent specialized in drafting research proposals, hypothesis generation, and contextualizing with literature."""

    def __init__(self,
                 name: str = "Researcher Agent",
                 model: str = "llama3-70b-8192",
                 temperature: float = 0.6,
                 max_tokens: int = 2000,
                 provider: str = "groq",
                 **kwargs):

        description = (
            "I specialize in generating academic research proposals, hypotheses, methodologies, and "
            "structured exploration of scientific questions."
        )

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

        researcher_prompt = """
As a Researcher Agent, your primary role is to help users with scientific research.

When responding:
1. Break down complex research topics into manageable components
2. Cite relevant literature and established methodologies
3. Generate clear, original hypotheses and suggest experimental designs
4. Identify potential limitations, assumptions, and challenges
5. Suggest follow-up research directions to expand the inquiry
6. Provide structured sections: Title, Background, Problem Statement, Objectives, Methodology, Expected Outcomes, Challenges
7. Use markdown formatting with proper headings
8. Be precise, academic, and scientifically grounded
"""
        self.update_system_prompt(self.system_prompt + researcher_prompt)

    async def run(self, user_query: str) -> str:
        """
        Main entry for producing a research document or idea expansion.

        Args:
            user_query: A topic, question, or goal to base research on

        Returns:
            Structured markdown-formatted research proposal
        """
        check = self.guardrails.check_input(user_query)
        if not check["passed"]:
            return f"🚫 Input blocked: {check['message']}"

        prompt = self._preprocess_query(user_query)
        result = await self.process(prompt)
        clean_response = self.guardrails.sanitize_output(result.get("response", ""))
        return self._postprocess_response(clean_response)

    def _preprocess_query(self, query: str) -> str:
        """
        Prepare user query for research proposal generation.

        Args:
            query: User input

        Returns:
            Full prompt
        """
        return f"""
Generate a full academic research proposal or structured exploration based on the following input:

\"\"\"{query}\"\"\"

Please organize your response using markdown headings and include the following sections:

- Title
- Introduction and Background
- Problem Statement
- Research Objectives or Hypotheses
- Proposed Methodology
- Expected Outcomes
- Potential Challenges or Limitations
- Follow-up Research Suggestions
- References (if applicable)
"""

    def _postprocess_response(self, response: str) -> str:
        """
        Post-format the LLM output for clarity and completeness.

        Args:
            response: Raw LLM response

        Returns:
            Clean markdown
        """
        if not response.strip():
            return "⚠️ No research output was generated."

        if not response.lower().startswith("#"):
            response = "# 🧪 Research Proposal\n\n" + response

        return response


# For direct testing
if __name__ == "__main__":
    import asyncio

    async def test_researcher():
        agent = ResearcherAgent()
        topic = "Investigating the impact of AI-driven personalized education on student performance"
        result = await agent.run(topic)
        print(result)

    asyncio.run(test_researcher())