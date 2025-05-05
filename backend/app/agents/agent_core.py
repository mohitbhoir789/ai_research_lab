"""
Agent Core Module - Minimalistic MVP Version

This module provides a simplified base agent with:
- Basic state management
- Conversation history tracking
- Error handling
"""
import asyncio
import logging
import uuid
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Minimalistic base agent that handles basic conversation and LLM interaction.
    """
    
    def __init__(
        self,
        model: str = "llama3-70b-8192",
        provider: str = "groq",
        agent_id: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            model: LLM model to use
            provider: LLM provider service
            agent_id: Optional unique ID for this agent instance
        """
        self.model = model
        self.provider = provider
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Agent state: 'ready', 'processing', or 'error'
        self.state = "ready"
        
        # Basic conversation history: list of {role, content} dicts
        self.conversation_history = []
        
        logger.info(f"Initialized agent {self.agent_id} with {model} on {provider}")
    
    def add_message(self, role: str, content: str) -> Dict[str, Any]:
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('system', 'user', or 'assistant')
            content: Message content
            
        Returns:
            The message that was added
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        self.conversation_history.append(message)
        return message
    
    async def run(self, user_input: str) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: Input from the user
            
        Returns:
            Agent's response
        """
        # Update state
        self.state = "processing"
        
        try:
            # Add user message to history
            self.add_message("user", user_input)
            
            # Generate response (to be implemented by child classes)
            response = await self._generate_response(user_input)
            
            # Add assistant message to history
            self.add_message("assistant", response)
            
            # Reset state
            self.state = "ready"
            
            return response
            
        except Exception as e:
            # Handle error
            logger.error(f"Error in agent {self.agent_id}: {str(e)}")
            self.state = "error"
            return f"I encountered an error: {str(e)}"
    
    async def _generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.
        To be implemented by child classes.
        
        Args:
            user_input: Input from the user
            
        Returns:
            Generated response
        """
        # This is a placeholder - subclasses should override this
        return "Base agent received: " + user_input
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary with agent state information
        """
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "model": self.model,
            "provider": self.provider,
            "message_count": len(self.conversation_history)
        }
    
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        """
        # Keep system messages
        system_messages = [
            msg for msg in self.conversation_history 
            if msg["role"] == "system"
        ]
        
        # Clear history and add back system messages
        self.conversation_history = system_messages.copy()
        
        # Reset state
        self.state = "ready"
        
        logger.info(f"Reset agent {self.agent_id}")


class LLMAgent(BaseAgent):
    """
    Agent that uses an LLM to generate responses.
    """
    
    def __init__(
        self,
        model: str = "llama3-70b-8192",
        provider: str = "groq",
        agent_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        guardrails: Optional[Any] = None
    ):
        """
        Initialize the LLM agent.
        
        Args:
            model: LLM model to use
            provider: LLM provider service
            agent_id: Optional unique ID for this agent
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            guardrails: Optional guardrails checker
        """
        super().__init__(model, provider, agent_id)
        
        from backend.app.utils.llm import LLMHandler
        # LLM parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.guardrails = guardrails
        
        self.llm = LLMHandler()
        
        # Add system prompt if provided
        if system_prompt:
            self.add_message("system", system_prompt)
    
    async def _generate_response(self, user_input: str) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            user_input: Input from the user
            
        Returns:
            Generated response
        """
        llm = self.llm
        
        # Get recent message history (last 10 messages)
        messages = self.conversation_history[-10:]
        
        # Format messages for LLM API
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        from backend.app.utils.llm import LLMConfig, LLMProvider
        config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        try:
            response, _ = await llm.generate(
                prompt=None,
                config=config,
                extra_params={"messages": formatted_messages}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            return "I'm having trouble generating a response right now."


class AgentManager:
    """
    Manages multiple agents and routes requests to the active agent.
    """
    
    def __init__(self):
        """Initialize the agent manager."""
        self.agents = {}
        self.active_agent_id = None
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the manager.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.agent_id] = agent
        
        # If this is the first agent, make it active
        if self.active_agent_id is None:
            self.active_agent_id = agent.agent_id
    
    def set_active_agent(self, agent_id: str) -> bool:
        """
        Set an agent as the active agent.
        
        Args:
            agent_id: ID of the agent to set as active
            
        Returns:
            Boolean indicating success
        """
        if agent_id in self.agents:
            self.active_agent_id = agent_id
            return True
        return False
    
    def get_active_agent(self) -> Optional[BaseAgent]:
        """
        Get the currently active agent.
        
        Returns:
            Active agent or None if no active agent
        """
        if self.active_agent_id is None:
            return None
        return self.agents.get(self.active_agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.
        
        Returns:
            List of agent information dictionaries
        """
        return [
            {
                "id": agent.agent_id,
                "type": type(agent).__name__,
                "state": agent.state,
                "is_active": agent.agent_id == self.active_agent_id
            }
            for agent_id, agent in self.agents.items()
        ]


class Memory:
    """
    Simple memory store for agents.
    """
    
    def __init__(self):
        """Initialize the memory store."""
        self.data = {}
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value in memory.
        
        Args:
            key: Key to store the value under
            value: Value to store
        """
        self.data[key] = value
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from memory.
        
        Args:
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Retrieved value or default
        """
        return self.data.get(key, default)
    
    def clear(self) -> None:
        """Clear all stored memory."""
        self.data = {}


class Tool:
    """
    Simple tool interface for agents to use.
    """
    
    def __init__(self, name: str, description: str, func: callable):
        """
        Initialize a tool.
        
        Args:
            name: Tool name
            description: Tool description
            func: Function to call when tool is used
        """
        self.name = name
        self.description = description
        self.func = func
    
    async def run(self, *args, **kwargs) -> Any:
        """
        Run the tool function.
        
        Returns:
            Result of the tool function
        """
        try:
            if asyncio.iscoroutinefunction(self.func):
                return await self.func(*args, **kwargs)
            else:
                return self.func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error running tool {self.name}: {str(e)}")
            return f"Error: {str(e)}"