import sys, os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
"""
Summarizer Agent Module
Specialized agent for summarizing text content while maintaining accuracy and clarity.
"""
import sys
from backend.app.agents.agent_core import BaseAgent
from backend.app.utils.llm import LLMConfig, LLMProvider

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SummarizerAgent(BaseAgent):
    """Agent specialized in creating clear, concise, and accurate summaries of content."""
    
    def __init__(
        self,
        name: str = "Summarizer Agent",
        model: str = "mixtral-8x7b-32768",  # Updated to use a valid Groq model
        temperature: float = 0.5,  # Lower temperature for more factual responses
        max_tokens: int = 1500,
        provider: str = "groq",
        guardrails=None,
        memory_manager=None,
        agent_id=None,
        **kwargs
    ):
        from backend.app.utils.guardrails import GuardrailsChecker
        from backend.app.utils.llm import LLMHandler
        from backend.app.mcp.mcp_protocol import MessageRole
        
        self.guardrails = guardrails or GuardrailsChecker()
        self.llm = LLMHandler()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_manager = memory_manager

        # Initialize base agent
        super().__init__(
            model=model,
            provider=provider,
            agent_id=agent_id
        )

        # Add specialized system prompt for the summarizer agent
        specialized_prompt = """
        As a Summarizer Agent, I excel at distilling complex information into clear, concise summaries.
        
        When summarizing content:
        1. I identify and prioritize the most important information, concepts, and findings
        2. I maintain factual accuracy and the original intent of the source material
        3. I organize information logically with clear structure using appropriate headings and formatting
        4. I remove redundancy and unnecessary details while preserving key context
        5. I use plain language to make complex topics accessible without oversimplifying
        6. I include proper attribution when summarizing specific sources or research
        
        I can summarize various types of content including research papers, articles, conversations, 
        concepts, theories, and experimental findings. I adapt my summarization approach based on the 
        content type and the user's specific needs.
        """
        
        self.add_message(MessageRole.SYSTEM, specialized_prompt)
        
    async def run(self, query: str) -> str:
        """
        Main entry point for running the agent with a query.
        
        Args:
            query: The user's input query or content to summarize
            
        Returns:
            A summarized version of the content
        """
        # Preprocess the query to enhance summarization quality
        processed_query = self._preprocess_query(query)
        
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Process the query through the LLM with optional guardrails
        if self.guardrails:
            # Check input safety first
            check_result = self.guardrails.check_input(processed_query)
            if not check_result["passed"]:
                return f"Error: {check_result['message']}"
            
            # Generate response
            response, _ = await self.llm.generate(prompt=processed_query, config=config)
            
            # Sanitize the output
            sanitized_response = self.guardrails.sanitize_output(response)
            final_response = self._postprocess_response(sanitized_response)
        else:
            response, _ = await self.llm.generate(prompt=processed_query, config=config)
            final_response = self._postprocess_response(response)
        
        # Return the final summarized response
        return final_response
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to guide the summarization.
        
        Args:
            query: The original query or content to summarize
            
        Returns:
            The processed query with summarization guidance
        """
        # Check if the query already contains summarization instructions
        if "summarize" in query.lower() or "summary" in query.lower():
            return query
        
        # If it's just content without clear instructions, add summarization prompt
        if len(query.split()) > 50:  # If it's a longer text, assume it needs summarization
            return f"""Please summarize the following content concisely while preserving all key information:

{query}

Provide the summary in a clear, structured format using markdown headings where appropriate."""
        
        # For shorter queries, assume it's a request about summarization
        return query
    
    def _postprocess_response(self, response: str) -> str:
        """
        Format the summary response for readability.
        
        Args:
            response: The raw LLM response
            
        Returns:
            Properly formatted and structured summary
        """
        # Ensure the response has proper markdown formatting
        if not any(marker in response for marker in ["#", "##", "###", "- ", "1. "]):
            # Add a title if none exists
            response = "# Summary\n\n" + response
        
        # Ensure there's a summary label if it's not already present
        if not response.lower().startswith(("# summary", "#summary", "summary:")):
            if not response.startswith("#"):
                response = "# Summary\n\n" + response
        
        return response
    
    async def summarize_with_focus(self, content: str, focus_areas: list, max_length: int = 500) -> str:
        """
        Specialized method for summarizing with specific focus areas.
        
        Args:
            content: The content to summarize
            focus_areas: List of specific topics or aspects to focus on
            max_length: Maximum length of the summary in words
            
        Returns:
            A focused summary of the content
        """
        focus_str = ", ".join(focus_areas)
        prompt = f"""Please summarize the following content in approximately {max_length} words or less.
Focus specifically on these aspects: {focus_str}.

{content}"""
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        if self.guardrails:
            # Check input safety first
            check_result = self.guardrails.check_input(prompt)
            if not check_result["passed"]:
                return f"Error: {check_result['message']}"
            
            # Generate response
            response, _ = await self.llm.generate(prompt=prompt, config=config)
            
            # Sanitize the output
            sanitized_response = self.guardrails.sanitize_output(response)
            return self._postprocess_response(sanitized_response)
        else:
            response, _ = await self.llm.generate(prompt=prompt, config=config)
            return self._postprocess_response(response)
    
    async def compare_and_summarize(self, contents: list, comparison_aspects: list = None) -> str:
        """
        Compare multiple pieces of content and summarize the similarities and differences.
        
        Args:
            contents: List of content strings to compare
            comparison_aspects: Optional list of specific aspects to compare
            
        Returns:
            A comparative summary
        """
        if len(contents) < 2:
            return "Need at least two pieces of content to compare."
        
        aspects_str = ""
        if comparison_aspects:
            aspects_str = f"Focus on comparing these specific aspects: {', '.join(comparison_aspects)}."
            
        formatted_contents = "\n\n---\n\nContent ".join([f"{i+1}:\n{content}" for i, content in enumerate(contents)])
        
        prompt = f"""Compare and summarize the following pieces of content. Identify key similarities and differences.
{aspects_str}

Content 1:
{formatted_contents}

Provide a structured comparative summary using markdown formatting."""
        
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if self.guardrails:
            # Check input safety first
            check_result = self.guardrails.check_input(prompt)
            if not check_result["passed"]:
                return f"Error: {check_result['message']}"
            
            # Generate response
            response, _ = await self.llm.generate(prompt=prompt, config=config)
            
            # Sanitize the output
            sanitized_response = self.guardrails.sanitize_output(response)
            return self._postprocess_response(sanitized_response)
        else:
            response, _ = await self.llm.generate(prompt=prompt, config=config)
            return self._postprocess_response(response)


# For testing the agent directly
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        agent = SummarizerAgent()
        
        # Test basic summarization
        test_content = """
        Transformer neural networks, introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, 
        revolutionized natural language processing. They rely on a self-attention mechanism to weigh the importance 
        of different words in a sequence. Unlike previous RNN-based models, transformers process entire sequences 
        in parallel rather than sequentially, making them more efficient for training on large datasets. 
        The architecture consists of encoders and decoders, with each containing self-attention and feed-forward 
        neural network layers. Transformers form the foundation of models like BERT, GPT, T5, and others that have 
        achieved state-of-the-art results in various NLP tasks. Their ability to capture long-range dependencies 
        and contextual relationships between words has made them the dominant architecture in modern NLP systems.
        """
        
        response = await agent.run(test_content)
        print("Basic Summarization Test:")
        print(response)
        print("\n" + "-"*50 + "\n")
        
        # Test focused summarization
        focused_response = await agent.summarize_with_focus(
            test_content, 
            ["attention mechanism", "advantages over RNNs"]
        )
        print("Focused Summarization Test:")
        print(focused_response)
    
    asyncio.run(test_agent())