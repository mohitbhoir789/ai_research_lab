# backend/app/agents/experimental_agent.py
from app.utils.llm import LLMHandler
from typing import Dict

class ExperimentalAgent:
    def __init__(self):
        self.llm = LLMHandler()
        self.model = "groq"
        self.persona = "Creative Experimental Researcher - Grounded but Innovative"

    async def run(self, input_text: str) -> str:
        """Run experimental idea generation."""
        inputs = {"input": input_text, "previous_context": ""}
        result = await self.ainvoke(inputs)
        return result["output"]
    
    async def ainvoke(self, inputs: Dict) -> Dict:
        """Suggest new hypotheses or experimental ideas."""
        user_query = inputs.get("input", "")
        previous_context = inputs.get("previous_context", "")

        prompt = f"""
You are an experimental AI researcher specializing in Computer Science and Data Science.

You have the following background context:

{previous_context}

The user's query is:

"{user_query}"

✅ Your task:
- Suggest 2–3 new **hypotheses** or **experimental ideas** based on the topic.
- Each hypothesis must be **logical**, **specific**, and **scientifically valid**.
- Avoid random speculation. Justify each idea briefly (1–2 lines why it's plausible).

Respond in Markdown, like:

### New Hypotheses:
1. **Hypothesis Title**: Description + Justification
2. **Hypothesis Title**: Description + Justification
3. **Hypothesis Title**: Description + Justification
"""
        response = await self.llm.generate(prompt, model=self.model)
        return {"output": response.strip()}