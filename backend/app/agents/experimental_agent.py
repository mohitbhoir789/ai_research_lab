"""
Experimental Agent Module
Specialized agent for designing detailed and methodologically sound experiments.
"""

import logging
from app.agents.agent_core import LLMAgent
from app.utils.guardrails import GuardRailsChecker

logger = logging.getLogger(__name__)


class ExperimentalAgent(LLMAgent):
    """Agent specialized in designing structured, testable, and rigorous experimental protocols."""

    def __init__(self,
                 name: str = "Experimental Agent",
                 model: str = "llama3-70b-8192",
                 temperature: float = 0.4,
                 max_tokens: int = 1800,
                 provider: str = "groq",
                 **kwargs):

        description = (
            "I specialize in designing clear, structured, and reproducible experimental protocols "
            "that align with research goals and hypotheses."
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

        specialized_prompt = """
As an Experimental Agent, your job is to translate research problems into well-defined experimental designs.

For each query, your output must follow this structure:

1. **Hypothesis**: A testable prediction with clear independent/dependent variables.
2. **Objectives**: What the experiment aims to demonstrate or validate.
3. **Experimental Setup**:
   - Population/Sample
   - Materials or tools required
   - Conditions (control vs experimental, blinding, randomization)
4. **Procedures**: Step-by-step experimental protocol.
5. **Data Collection Methods**: What will be measured, how, and how often.
6. **Analysis Plan**:
   - Statistical tests
   - Evaluation metrics
   - Software/tools for analysis
7. **Expected Outcomes**: What results are anticipated and what they might imply.
8. **Risks & Mitigations**: Ethical, technical, or logistical risks and how to address them.

Use markdown formatting with bold or headings. Be concise but thorough. Include assumptions and justifications where relevant.
"""
        self.update_system_prompt(self.system_prompt + specialized_prompt)
        self.guardrails = GuardRailsChecker()

    async def run(self, query: str) -> str:
        """
        Generate a detailed experiment design based on a research question or hypothesis.

        Args:
            query: The research question or design request.

        Returns:
            A markdown-formatted experimental protocol.
        """
        input_check = self.guardrails.check_input(query)
        if not input_check["passed"]:
            return f"🚫 Blocked by guardrails: {input_check['message']}"

        prompt = self._preprocess_query(query)
        result = await self.process(prompt)
        output = result.get("response", "")
        return self.guardrails.sanitize_output(self._postprocess_response(output))

    def _preprocess_query(self, query: str) -> str:
        """
        Builds the full prompt to guide the experimental design process.

        Args:
            query: User-provided research topic or hypothesis.

        Returns:
            A formatted instruction prompt.
        """
        return f"""
You're an expert in experimental methodology. Based on the following topic or research goal, create a complete experiment plan.

\"\"\"{query}\"\"\"

Use all sections listed in your instructions and present the response in markdown format only.
"""

    def _postprocess_response(self, response: str) -> str:
        """
        Ensures output is clean, complete, and structured.

        Args:
            response: Raw LLM response.

        Returns:
            Markdown output string.
        """
        if not response.strip():
            return "⚠️ No experimental design was generated."

        if not response.lower().startswith("# experiment") and "# Experimental" not in response:
            response = "# Experimental Design\n\n" + response

        return response


# For standalone testing
if __name__ == "__main__":
    import asyncio

    async def test():
        agent = ExperimentalAgent()
        topic = "Design an experiment to test whether LLM-based tutors improve student comprehension of calculus."
        design = await agent.run(topic)
        print(design)

    asyncio.run(test())