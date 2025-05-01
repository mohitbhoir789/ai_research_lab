"""
Template Agent Module
This template shows how to create a specialized agent based on the LLMAgent class.
Copy this template for creating new agent types.
"""

from app.agents.agent_core import LLMAgent
import logging

logger = logging.getLogger(__name__)

class TemplateAgent(LLMAgent):
    """Template for creating specialized agents."""
    
    def __init__(self, 
                 name: str = "Template Agent", 
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 provider: str = "groq",
                 **kwargs):
        
        description = "I am a specialized agent template that can be customized for specific tasks."
        
        super().__init__(
            name=name,
            description=description,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            **kwargs
        )
        
        # Add specialized system prompt for this agent type
        specialized_prompt = """
        As a specialized agent, I focus on [describe specific functionality].
        
        When responding to queries:
        1. [First specialized behavior]
        2. [Second specialized behavior]
        3. [Third specialized behavior]
        4. [Fourth specialized behavior]
        5. [Fifth specialized behavior]
        """
        
        # Update the system prompt with specialized instructions
        self.update_system_prompt(self.system_prompt + specialized_prompt)
        
    async def run(self, query: str) -> str:
        """
        Main entry point for running the agent with a query.
        
        Args:
            query: The user's input query
            
        Returns:
            A string response from the agent
        """
        # Preprocess the query if needed
        processed_query = self._preprocess_query(query)
        
        # Process the query through the LLM
        result = await self.process(processed_query)
        
        # Postprocess the response if needed
        final_response = self._postprocess_response(result.get("response", ""))
        
        # Return the final response
        return final_response
    
    def _preprocess_query(self, query: str) -> str:
        """
        Optional preprocessing of the user's query.
        
        Args:
            query: The original query
            
        Returns:
            The processed query
        """
        # Add any preprocessing logic here
        # For example, adding context, reformatting, etc.
        return query
    
    def _postprocess_response(self, response: str) -> str:
        """
        Optional postprocessing of the LLM's response.
        
        Args:
            response: The raw LLM response
            
        Returns:
            The processed response
        """
        # Add any postprocessing logic here
        # For example, formatting, extracting specific information, etc.
        return response


# For testing the agent directly
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        agent = TemplateAgent()
        response = await agent.run("Test query for the template agent.")
        print(response)
    
    asyncio.run(test_agent())