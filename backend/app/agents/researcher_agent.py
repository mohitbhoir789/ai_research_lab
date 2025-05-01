# backend/app/agents/researcher_agent.py
from app.utils.llm import LLMHandler

class ResearcherAgent:
    def __init__(self):
        self.llm = LLMHandler()
        self.model = "groq"

    async def run(self, input_text: str) -> str:
        """Run research plan generation."""
        return await self.generate_research_plan(input_text)

    async def generate_research_plan(self, topic: str) -> str:
        prompt = f"""
You are an expert AI Research Scientist.

Create a detailed mini research proposal based on the following topic:

Topic: {topic}

Follow this structure:
1. **Research Question**: Formulate a clear, novel research question.
2. **Hypothesis**: A possible hypothesis based on known gaps or assumptions.
3. **Proposed Methodology**: 
    - Describe the method or model you would use.
    - Suggest specific datasets if relevant (e.g., Kaggle, OpenML, or real-world sources).
    - Mention what type of experiments you would conduct.
4. **Potential Challenges**: 
    - List 2–3 risks or difficult areas (e.g., data availability, computational cost, ethical issues).
5. **Expected Outcomes**: 
    - Describe what successful research might show.
    - Highlight possible impact on real-world applications or academic theory.

Write in a formal academic style. Output should be in Markdown format for clarity.
Focus on realistic, creative, and feasible research ideas grounded in computer science or data science fields.
"""
        response = await self.llm.generate(prompt, model=self.model)
        return response