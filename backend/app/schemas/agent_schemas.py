from pydantic import BaseModel
from typing import List, Optional

class RetrieverOutput(BaseModel):
    book_context: Optional[str]
    paper_context: Optional[str]
    wikipedia_context: Optional[str]

class ResearchPlan(BaseModel):
    hypothesis: str
    objectives: List[str]
    methodology: str

class ResearchOutput(BaseModel):
    expanded_research: str

class CritiqueOutput(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]

class VerificationOutput(BaseModel):
    verified_facts: List[str]
    missing_facts: Optional[List[str]]

class SummaryOutput(BaseModel):
    intro: Optional[str]
    key_concepts: Optional[List[str]]
    example_applications: Optional[List[str]]
    conclusion: Optional[str]