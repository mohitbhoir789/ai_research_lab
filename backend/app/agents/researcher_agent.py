"""
Researcher Agent Module

A specialized agent for research-related tasks that can:
- Generate research proposals
- Analyze research questions
- Provide literature reviews
- Suggest experiments
"""

import logging
from typing import Dict, List, Any, Optional

from backend.app.agents.agent_core import LLMAgent, Memory, Tool
from backend.app.utils.llm import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

class ResearcherAgent(LLMAgent):
    """
    An agent specialized in academic research, experiment design, and analysis.
    """
    
    def __init__(
        self,
        model: str = "llama3-70b-8192",
        provider: str = "groq",
        agent_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        guardrails=None
    ):
        """
        Initialize the researcher agent.
        
        Args:
            model: LLM model to use
            provider: LLM provider service
            agent_id: Optional unique ID for this agent
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            guardrails: Optional guardrails checker
        """
        # Define research-specific system prompt
        system_prompt = """
        You are an expert AI research assistant with expertise across multiple scientific domains.
        
        Your capabilities include:
        - Generating research proposals and hypotheses
        - Suggesting experimental designs and methodologies
        - Analyzing research questions from multiple perspectives
        - Providing structured literature reviews
        - Identifying gaps in current research
        - Suggesting novel approaches to complex problems
        
        When responding:
        - Be scientifically rigorous and evidence-based
        - Structure your responses clearly with appropriate headings
        - Note limitations and uncertainties where relevant
        - Suggest specific directions for further investigation
        - Use academic language appropriate for the field
        - When discussing experimental designs, include key variables, controls, and analysis methods
        
        Always maintain a scientific, objective approach while being helpful and thorough.
        """
        
        # Initialize the base LLMAgent
        super().__init__(
            model=model,
            provider=provider,
            agent_id=agent_id or "researcher",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize additional components
        self.memory = Memory()
        self.guardrails = guardrails
        
        # Track research domains and interests
        self.research_domains = set()
        self.last_research_topic = None
    
    async def generate_research_proposal(self, topic: str) -> str:
        """
        Generate a structured research proposal on a given topic.
        
        Args:
            topic: Research topic
            
        Returns:
            Research proposal as formatted text
        """
        proposal_prompt = f"""
        Create a comprehensive research proposal on: {topic}
        
        Include the following sections:
        1. Abstract
        2. Background & Literature Review
        3. Research Questions & Hypotheses
        4. Methodology
        5. Expected Results
        6. Implications & Impact
        7. References (key works only)
        
        Format your response using markdown with clear headings and bullet points where appropriate.
        """
        
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response, _ = await self.llm.generate(prompt=proposal_prompt, config=config)
        
        # Track this research domain
        self.research_domains.add(topic.split()[0])  # Simple domain tracking
        self.last_research_topic = topic
        self.memory.store("last_proposal_topic", topic)
        
        return response
    
    async def suggest_experiments(self, research_question: str) -> str:
        """
        Suggest experiments to answer a research question.
        
        Args:
            research_question: The research question
            
        Returns:
            Experimental suggestions as formatted text
        """
        experiment_prompt = f"""
        For the following research question: "{research_question}"
        
        Design 3 different experimental approaches that could help answer this question.
        
        For each experiment, include:
        1. Experimental design overview
        2. Independent and dependent variables
        3. Control conditions
        4. Required materials/equipment
        5. Key methodological considerations
        6. Analysis approach
        7. Potential limitations
        
        Format your response using markdown with clear headings for each experiment.
        """
        
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response, _ = await self.llm.generate(prompt=experiment_prompt, config=config)
        
        # Store in memory
        self.memory.store("last_research_question", research_question)
        
        return response
    
    async def analyze_limitations(self, proposal_or_experiment: str) -> str:
        """
        Analyze limitations of a research proposal or experimental design.
        
        Args:
            proposal_or_experiment: Text of proposal or experiment to analyze
            
        Returns:
            Analysis of limitations
        """
        limitations_prompt = f"""
        Analyze the following research proposal or experimental design for limitations
        and potential improvements:
        
        {proposal_or_experiment}
        
        Include in your analysis:
        1. Methodological limitations
        2. Potential confounding variables
        3. Statistical considerations
        4. Generalizability issues
        5. Practical implementation challenges
        6. Ethical considerations
        7. Suggested improvements for each limitation
        
        Format your response using markdown with clear headings and prioritize the most
        significant limitations first.
        """
        
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response, _ = await self.llm.generate(prompt=limitations_prompt, config=config)
        return response
    
    async def suggest_interdisciplinary_approaches(self, topic: str) -> str:
        """
        Suggest interdisciplinary approaches to a research topic.
        
        Args:
            topic: Research topic
            
        Returns:
            Interdisciplinary suggestions
        """
        interdisciplinary_prompt = f"""
        For the research topic: "{topic}"
        
        Suggest how this topic could be approached from 5 different disciplinary perspectives.
        For each perspective, explain:
        
        1. Key theories or frameworks from that discipline that could be applied
        2. Methodological approaches typical of that discipline
        3. New insights this disciplinary perspective might offer
        4. Potential for integration with other disciplinary approaches
        
        Consider diverse fields such as natural sciences, social sciences, humanities,
        engineering, medicine, arts, etc.
        
        Format your response using markdown with clear headings for each disciplinary approach.
        """
        
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response, _ = await self.llm.generate(prompt=interdisciplinary_prompt, config=config)
        return response
    
    async def run(self, user_input: str) -> str:
        """
        Process research-related user input.
        
        Args:
            user_input: Input from the user
            
        Returns:
            Research-focused response
        """
        self.state = "processing"
        
        try:
            # Add user message to history
            self.add_message("user", user_input)
            
            # Detect intent to determine specialized function
            if "research proposal" in user_input.lower():
                # Extract topic
                topic = user_input.replace("research proposal", "").strip()
                if not topic:
                    topic = "Please specify a research topic"
                
                response = await self.generate_research_proposal(topic)
                
            elif "suggest experiments" in user_input.lower() or "experimental design" in user_input.lower():
                # Extract research question
                question = user_input.replace("suggest experiments", "").replace("experimental design", "").strip()
                if not question:
                    question = "Please specify a research question"
                
                response = await self.suggest_experiments(question)
                
            elif "limitations" in user_input.lower() and "analyze" in user_input.lower():
                # Check if we should use previous proposal
                if "previous" in user_input.lower() and self.conversation_history:
                    # Find last assistant message
                    for msg in reversed(self.conversation_history):
                        if msg["role"] == "assistant":
                            proposal = msg["content"]
                            break
                    else:
                        proposal = "Please provide a proposal or experiment to analyze"
                else:
                    proposal = user_input
                
                response = await self.analyze_limitations(proposal)
                
            elif "interdisciplinary" in user_input.lower():
                # Extract topic
                topic = user_input.replace("interdisciplinary", "").strip()
                if not topic and self.last_research_topic:
                    topic = self.last_research_topic
                elif not topic:
                    topic = "Please specify a research topic"
                
                response = await self.suggest_interdisciplinary_approaches(topic)
                
            else:
                # Handle as general research question
                config = LLMConfig(
                    model=self.model,
                    provider=LLMProvider(self.provider),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                response, _ = await self.llm.generate(prompt=user_input, config=config)
            
            # Apply guardrails if available
            if self.guardrails:
                response = self.guardrails.sanitize_output(response)
            
            # Add response to history
            self.add_message("assistant", response)
            
            # Reset state
            self.state = "ready"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in researcher agent: {str(e)}")
            self.state = "error"
            return "I encountered an error while processing your research request. Please try a different query or provide more details."