# backend/app/agents/verifier_agent.py
from app.utils.llm import LLMHandler

class VerifierAgent:
    def __init__(self):
        self.llm = LLMHandler()
        self.model = "groq"  # You had gemini, but groq is your default now. We can add Gemini later cleanly.

    async def run(self, input_text: str) -> str:
        """Run verification check."""
        return await self.verify(input_text)
    
    async def verify(self, research_text: str) -> str:
        prompt = f"""
You are a fact-verification expert.

Verify the correctness of the following research summary.
List any false or questionable claims, and suggest corrections.

Research Text:x 
\"\"\"
{research_text}
\"\"\"
"""
        response = await self.llm.generate(prompt, model=self.model)
        return response