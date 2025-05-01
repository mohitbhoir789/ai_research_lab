# backend/app/agents/planner_agent.py
from app.utils.llm import LLMHandler

class PlannerAgent:
    def __init__(self):
        self.llm = LLMHandler()
        self.model = "groq"  # Define the model you want to use
    
    
    async def run(self, input_text: str) -> str:
        """Run planning directly."""
        return await self.plan(input_text)
    
    async def plan(self, topic: str) -> str:
        prompt = f"""
You are a research project planner.

Given the topic: "{topic}", create a 4-step research plan.
Each step should be concise and logically build toward understanding the topic fully.
"""
        response = await self.llm.generate(prompt, model=self.model)
        return response
    