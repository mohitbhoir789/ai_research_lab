"""
Critic Agent Module
Specialized agent for reviewing research proposals, methodologies, and ideas with a critical lens.
"""

import logging
from app.agents.agent_core import LLMAgent
from app.utils.guardrails import GuardRailsChecker

logger = logging.getLogger(__name__)


class CriticAgent(LLMAgent):
    """Agent specialized in providing critical analysis, constructive feedback, and scientific review."""

    def __init__(self,
                 name: str = "Critic Agent",
                 model: str = "llama3-70b-8192",
                 temperature: float = 0.4,
                 max_tokens: int = 1500,
                 provider: str = "groq",
                 **kwargs):

        description = (
            "I specialize in analyzing research proposals, scientific methods, and hypotheses. "
            "My focus is on identifying gaps, biases, and assumptions, and offering constructive feedback."
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

        critic_prompt = """
As a Critic Agent, your job is to analyze research proposals or concepts critically and constructively.

Follow this structure:
1. **Logical Consistency** – Are arguments well-supported and conclusions valid?
2. **Methodological Soundness** – Are the methods scientifically rigorous and reproducible?
3. **Assumptions & Biases** – Identify any flawed premises or unexplored biases.
4. **Feasibility** – Are the proposed experiments or steps practically achievable?
5. **Impact & Novelty** – Does the idea add something genuinely new or valuable?
6. **Constructive Suggestions** – Offer specific improvements or alternate approaches.

Format your response using markdown with clear section headers. Be professional, analytical, and helpful.
"""
        self.update_system_prompt(self.system_prompt + critic_prompt)

    async def run(self, input_text: str) -> str:
        """
        Run the critic analysis on the provided input (research proposal, plan, etc.)

        Args:
            input_text: The content to critique.

        Returns:
            Markdown-formatted critical review.
        """
        # Input guardrail
        check = self.guardrails.check_input(input_text)
        if not check["passed"]:
            return f"⚠️ Input blocked by guardrails: {check['message']}"

        prompt = self._preprocess_query(input_text)
        result = await self.process(prompt)

        # Output guardrail
        clean_response = self.guardrails.sanitize_output(result.get("response", ""))
        return self._postprocess_response(clean_response)

    def _preprocess_query(self, content: str) -> str:
        """
        Prepares the content for analysis.

        Args:
            content: Input content to critique.

        Returns:
            Prompt string.
        """
        return f"""
Critically evaluate the following research idea or proposal:

\"\"\"{content}\"\"\"

Use markdown sections and be detailed yet constructive. Highlight strengths and weaknesses, and suggest improvements.
"""

    def _postprocess_response(self, response: str) -> str:
        """
        Postprocesses the LLM response.

        Args:
            response: Raw LLM output.

        Returns:
            Cleaned markdown output.
        """
        if not response.strip():
            return "⚠️ No critique was generated."

        if not response.lower().startswith("# critique"):
            response = "# Critique & Review\n\n" + response

        return response


# For direct testing
if __name__ == "__main__":
    import asyncio

    async def test_critic():
        agent = CriticAgent()
        proposal = """
        We propose building a quantum machine learning model that operates on entangled qubits to forecast stock prices.
        The system will use Bell pair correlations to reduce uncertainty and produce 95% accurate predictions.
        """
        result = await agent.run(proposal)
        print(result)

    asyncio.run(test_critic())