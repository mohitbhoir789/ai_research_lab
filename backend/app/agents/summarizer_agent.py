# backend/app/agents/summarizer_agent.py

from app.utils.llm import LLMHandler

class SummarizerAgent:
    def __init__(self):
        self.llm = LLMHandler()
        self.model = "groq"  # Default model, you can customize later

    async def summarize(self, topic: str, style: str = "detailed", additional_context: str = None) -> str:
        """Smart summarization based on style (simple or detailed)."""

        if style == "simple":
            prompt = f"""
You are an expert AI instructor.

Write a **short one-paragraph** clear explanation about:

Topic: {topic}

✅ Max 5–6 lines, no headings.
✅ University-level clarity, no jargon.

{f"Additional Context:\n{additional_context}" if additional_context else ""}
"""
        else:
            prompt = f"""
You are an expert AI instructor.

Write a **detailed structured explanation** about:

Topic: {topic}

✅ Structure:
1. Introduction (3–4 lines)
2. Key Concepts (5–7 bullet points)
3. Example Applications (2–3 examples)
4. Conclusion (2–3 lines)

✅ Markdown format.
{f"Additional Context:\n{additional_context}" if additional_context else ""}
"""

        response = await self.llm.generate(prompt, model=self.model)
        return response

    async def run(self, topic: str, style: str = "detailed", additional_context: str = None):
        return await self.summarize(topic, style=style, additional_context=additional_context)