"""
Verifier Agent Module
Specialized agent for validating claims, checking factual accuracy, and assessing feasibility.
"""

import logging
from backend.app.agents.agent_core import LLMAgent
from backend.app.utils.guardrails import GuardrailsChecker
from backend.app.utils.llm import LLMConfig, LLMProvider
from typing import Optional

logger = logging.getLogger(__name__)


class VerifierAgent(LLMAgent):
    """Agent specialized in verifying the factual, logical, and practical validity of research outputs."""

    def __init__(self,
                 model: str = "llama3-70b-8192",
                 provider: str = "groq",
                 agent_id: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1200,
                 **kwargs):

        specialized_prompt = """
As a Verifier Agent, you rigorously evaluate information and outputs for:

1. **Factual Accuracy**: Validate data, dates, citations, or claims based on known sources.
2. **Logical Consistency**: Identify contradictions, gaps in reasoning, or flawed assumptions.
3. **Scientific Feasibility**: Assess whether the experiment, methodology, or result is practically viable.
4. **Ethical & Risk Assessment**: Flag unethical practices, unrealistic expectations, or overlooked risks.

When verifying, structure your output with:

- ✅ Confirmed Facts
- ⚠️ Detected Issues
- 🔍 Recommendations / Corrections
- 🧪 Feasibility & Risk Analysis

Respond using markdown formatting. If additional context is needed, clearly note limitations. Always aim for objective, critical evaluation.
"""
        super().__init__(
            model=model,
            provider=provider,
            agent_id=agent_id,
            system_prompt=specialized_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        self.guardrails = GuardrailsChecker()

    async def run(self, content_to_verify: str) -> str:
        """
        Analyze and verify a given content block (summary, proposal, etc.)

        Args:
            content_to_verify: The research proposal, claim, or summary to evaluate.

        Returns:
            A structured verification report.
        """
        # Guardrails input validation
        check = await self.guardrails.check_input(content_to_verify)
        if not check["passed"]:
            return f"🚫 Input rejected: {check['reason']}"

        prompt = self._preprocess_query(content_to_verify)
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response, _ = await self.llm.generate(prompt=prompt, config=config)
        raw_response = response

        # Guardrails output sanitization
        return self.guardrails.sanitize_output(self._postprocess_response(raw_response))

    def _preprocess_query(self, content: str) -> str:
        """
        Build the full verification prompt.

        Args:
            content: Content to verify

        Returns:
            Formatted instruction string.
        """
        return f"""
Please critically verify the following content:

\"\"\"{content}\"\"\"

Use markdown headings and the structure in your instructions. Confirm facts, flag issues, and provide expert-level verification.
"""

    def _postprocess_response(self, response: str) -> str:
        """
        Ensure markdown structure and tagging in final output.

        Args:
            response: Raw LLM output

        Returns:
            Cleaned and structured response.
        """
        if not response.strip():
            return "⚠️ No verification results generated."

        if not response.lower().startswith("# verification"):
            response = "# Verification Report\n\n" + response

        return response


# For direct testing
if __name__ == "__main__":
    import asyncio

    async def test():
        agent = VerifierAgent()
        sample_claim = """
        Our experiment claims that GPT-4 can consistently outperform human experts in solving undergraduate physics problems with 98% accuracy. 
        We used 50 manually selected problems and evaluated the answers qualitatively.
        """
        output = await agent.run(sample_claim)
        print(output)

    asyncio.run(test())