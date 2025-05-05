"""
Specialized Agent Template

This file provides a template for creating specialized agents
that inherit from the base LLMAgent class.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

# Import the base agent classes
from app.agents.agent_core import LLMAgent, Memory, Tool

logger = logging.getLogger(__name__)

class SpecializedAgent(LLMAgent):
    """
    Template for a specialized agent that performs a specific task.
    
    This agent extends the base LLMAgent with:
    - Domain-specific system prompt
    - Custom pre-processing of user input
    - Custom post-processing of LLM output
    - Specialized tools for its domain
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
        Initialize the specialized agent.
        
        Args:
            model: LLM model to use
            provider: LLM provider service
            agent_id: Optional unique ID for this agent
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            guardrails: Optional guardrails checker
        """
        # Define the domain-specific system prompt
        system_prompt = """
        You are a specialized AI assistant that helps with [specific domain].
        Your goal is to [describe the agent's main purpose and how it should respond].
        
        When answering:
        - [Specific instruction 1]
        - [Specific instruction 2]
        - [Specific instruction 3]
        
        Always provide [type of information] in your responses.
        """
        
        # Initialize the base LLMAgent
        super().__init__(
            model=model,
            provider=provider,
            agent_id=agent_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Additional specialized components
        self.memory = Memory()
        self.guardrails = guardrails
        self.tools = self._initialize_tools()
        
        # Domain-specific state
        self.domain_state = {
            "last_topics": [],
            "confidence_level": "high",
            "specialized_data": {}
        }
    
    def _initialize_tools(self) -> Dict[str, Tool]:
        """
        Initialize domain-specific tools for this agent.
        
        Returns:
            Dictionary of tool name to Tool object
        """
        tools = {}
        
        # Example tool: Data processor
        tools["process_data"] = Tool(
            name="process_data",
            description="Process domain-specific data",
            func=self._process_data
        )
        
        # Example tool: Generate visualization
        tools["generate_visualization"] = Tool(
            name="generate_visualization",
            description="Generate a visualization of the data",
            func=self._generate_visualization
        )
        
        return tools
    
    async def _process_data(self, data: Any) -> Dict[str, Any]:
        """
        Process domain-specific data.
        
        Args:
            data: The data to process
            
        Returns:
            Processed data results
        """
        # Implement domain-specific data processing logic
        try:
            # Example processing
            results = {
                "processed": True,
                "summary": f"Processed {len(data)} items" if isinstance(data, list) else "Processed data",
                "key_findings": ["Finding 1", "Finding 2"]
            }
            
            # Store in memory for later use
            self.memory.store("last_processed_data", results)
            
            return results
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_visualization(self, data_key: str) -> str:
        """
        Generate a visualization of stored data.
        
        Args:
            data_key: Key of data in memory to visualize
            
        Returns:
            Description or URL of visualization
        """
        # Example visualization generation
        data = self.memory.retrieve(data_key)
        if not data:
            return "No data found for visualization"
        
        # In a real implementation, this might generate charts or graphs
        return f"Generated visualization for {data_key} (mock implementation)"
    
    async def _preprocess_input(self, user_input: str) -> str:
        """
        Preprocess user input before sending to LLM.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Preprocessed input
        """
        # Example preprocessing:
        # 1. Extract key terms
        # 2. Add context from memory if relevant
        # 3. Format appropriately for the domain
        
        # Extract potential topics
        topics = self._extract_topics(user_input)
        self.domain_state["last_topics"] = topics
        
        # Add context from memory if available
        enhanced_input = user_input
        if topics and any(self.memory.retrieve(f"topic_{topic}") for topic in topics):
            context = "\n\nRelevant context from previous interactions:\n"
            for topic in topics:
                topic_data = self.memory.retrieve(f"topic_{topic}")
                if topic_data:
                    context += f"- {topic}: {topic_data}\n"
            enhanced_input += context
        
        # Apply domain-specific formatting if needed
        # (Example: if this was a SQL agent, it might format SQL queries)
        
        return enhanced_input
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted topics
        """
        # This is a placeholder - in a real implementation this might use:
        # - Simple keyword matching
        # - NLP-based entity extraction
        # - Topic modeling
        
        # Simple example implementation
        common_topics = ["research", "data", "analysis", "visualization", "methodology"]
        found_topics = [topic for topic in common_topics if topic.lower() in text.lower()]
        return found_topics
    
    async def _postprocess_output(self, output: str) -> str:
        """
        Postprocess LLM output before returning to user.
        
        Args:
            output: Raw LLM output
            
        Returns:
            Processed output
        """
        # Example postprocessing:
        # 1. Add citations if needed
        # 2. Format consistently
        # 3. Add domain-specific enhancements
        
        # Store topics from this interaction for future reference
        for topic in self.domain_state["last_topics"]:
            topic_summary = f"Discussed in context: {output[:100]}..."
            self.memory.store(f"topic_{topic}", topic_summary)
        
        # Example: Add a confidence indicator
        if self.domain_state["confidence_level"] == "high":
            processed_output = output
        else:
            disclaimer = "\n\nNote: This response is based on limited information and may require further verification."
            processed_output = output + disclaimer
        
        # Apply any guardrails if set
        if self.guardrails:
            processed_output = self.guardrails.sanitize_output(processed_output)
        
        return processed_output
    
    async def run(self, user_input: str) -> str:
        """
        Override the base run method to add pre/post processing.
        
        Args:
            user_input: Input from the user
            
        Returns:
            Agent's response
        """
        self.state = "processing"
        
        try:
            # Preprocess the input
            processed_input = await self._preprocess_input(user_input)
            
            # Add user message to history (use the original input for history)
            self.add_message("user", user_input)
            
            # Check if a tool should be used
            tool_name = self._detect_tool_need(processed_input)
            if tool_name and tool_name in self.tools:
                # Extract tool parameters
                params = self._extract_tool_params(processed_input, tool_name)
                
                # Run the tool
                tool_result = await self.tools[tool_name].run(**params)
                
                # Generate response incorporating tool results
                response = await self._generate_response_with_tool_results(
                    processed_input, tool_name, tool_result
                )
            else:
                # Generate standard response
                response = await self._generate_response(processed_input)
            
            # Postprocess the output
            final_response = await self._postprocess_output(response)
            
            # Add assistant message to history
            self.add_message("assistant", final_response)
            
            # Reset state
            self.state = "ready"
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in specialized agent: {str(e)}")
            self.state = "error"
            return f"I encountered an error processing your request: {str(e)}"
    
    def _detect_tool_need(self, user_input: str) -> Optional[str]:
        """
        Detect if a tool should be used based on user input.
        
        Args:
            user_input: Processed user input
            
        Returns:
            Tool name if a tool should be used, None otherwise
        """
        # Placeholder implementation - in a real agent this might:
        # - Use regex pattern matching
        # - Use LLM to classify the request
        # - Use keyword detection
        
        if "process data" in user_input.lower():
            return "process_data"
        elif "visualization" in user_input.lower():
            return "generate_visualization"
        
        return None
    
    def _extract_tool_params(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameters for a tool from user input.
        
        Args:
            user_input: Processed user input
            tool_name: The tool to extract parameters for
            
        Returns:
            Dictionary of parameter name to value
        """
        # Placeholder implementation - in a real agent this would be:
        # - More sophisticated parameter extraction
        # - Potentially using LLM to help extract parameters
        
        params = {}
        
        if tool_name == "process_data":
            # Example: Extract data from input
            params["data"] = user_input  # Simplified
            
        elif tool_name == "generate_visualization":
            # Example: Use most recent data by default
            params["data_key"] = "last_processed_data"
        
        return params
    
    async def _generate_response_with_tool_results(
        self, user_input: str, tool_name: str, tool_result: Any
    ) -> str:
        """
        Generate a response that incorporates tool results.
        
        Args:
            user_input: Processed user input
            tool_name: Name of the tool that was used
            tool_result: Results from the tool
            
        Returns:
            Generated response
        """
        # Create a prompt that includes the tool results
        tool_prompt = f"""
        The user asked: {user_input}
        
        I used the {tool_name} tool, which returned:
        {tool_result}
        
        Based on this tool output, provide a helpful response to the user.
        """
        from app.utils.llm import LLMConfig, LLMProvider
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response, _ = await self.llm.generate(
            prompt=tool_prompt,
            config=config
        )
        return response


# Example instantiation and usage
async def example_usage():
    # Create the specialized agent
    agent = SpecializedAgent(
        model="llama3-70b-8192",
        provider="groq",
        temperature=0.7
    )
    
    # Run the agent with a user query
    response = await agent.run("Can you help me analyze this data and create a visualization?")
    print(response)


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())