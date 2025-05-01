# backend/app/agents/critic_agent.py
from app.utils.llm import LLMHandler

class CriticAgent:
    def __init__(self):
        self.llm = LLMHandler()
        self.model = "groq"

    async def run(self, input_text: str) -> str:
        """Run critique on given input."""
        return await self.critique_research_plan(input_text)

    async def critique_research_plan(self, research_plan: str) -> str:
        prompt = f"""
You are a senior AI Research Peer Reviewer.

Critique the following research proposal:

---
{research_plan}
---

Your review should include:
1. **Strengths**: What is good about this research idea?
2. **Weaknesses**: Any flaws, missing elements, unrealistic assumptions?
3. **Suggestions for Improvement**: How can this plan be made better?
4. **Critique Score** (optional): A rough score from 1 (poor) to 10 (excellent).

Write in a formal academic tone. Output using Markdown format for readability.
"""
        response = await self.llm.generate(prompt, model=self.model)
        return response