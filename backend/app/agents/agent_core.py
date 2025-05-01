"""
Agent Core Module
Core classes and utilities for agents: Memory, Tool, BaseAgent, LLMAgent, AgentManager.
Refactored to match the structure from the paste.txt example.
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio
import re

from app.utils.llm import LLMHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Memory:
    """Basic memory component for agents to store and retrieve information."""
    
    def __init__(self):
        self.short_term: Dict[str, Any] = {}
        self.long_term: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
    
    def add_to_short_term(self, key: str, value: Any) -> None:
        """Add information to short-term memory."""
        self.short_term[key] = value
    
    def get_from_short_term(self, key: str) -> Any:
        """Retrieve information from short-term memory."""
        return self.short_term.get(key)
    
    def add_to_long_term(self, information: Dict[str, Any]) -> None:
        """Add information to long-term memory."""
        information['timestamp'] = datetime.now().isoformat()
        self.long_term.append(information)
    
    def search_long_term(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search long-term memory based on query parameters."""
        results = []
        for item in self.long_term:
            match = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                results.append(item)
        return results
    
    def add_to_conversation(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_conversation_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve conversation history, optionally limited to the most recent messages."""
        if max_messages:
            return self.conversation_history[-max_messages:]
        return self.conversation_history
    
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term = {}
    
    def save_state(self, file_path: str) -> None:
        """Save the memory state to a file."""
        with open(file_path, 'w') as f:
            json.dump({
                'short_term': self.short_term,
                'long_term': self.long_term,
                'conversation_history': self.conversation_history
            }, f)
    
    def load_state(self, file_path: str) -> None:
        """Load the memory state from a file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.short_term = data.get('short_term', {})
                self.long_term = data.get('long_term', [])
                self.conversation_history = data.get('conversation_history', [])


class Tool:
    """A tool that agents can use to perform specific actions."""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool function with given arguments."""
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return {"error": str(e)}


class BaseAgent(ABC):
    """Base abstract class for all agents."""
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.memory = Memory()
        self.tools: Dict[str, Tool] = {}
        self.is_active = False
        self.handoff_queue: List[Tuple[str, Dict[str, Any]]] = []
        
        # Initialize system prompt
        self.system_prompt = f"""You are {name}, an AI assistant with the following description: {description}.
        Your goal is to assist users by providing helpful, accurate, and relevant information.
        """
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolbox."""
        self.tools[tool.name] = tool
        # Update system prompt to include tool information
        tools_description = '\n'.join([f"- {tool.name}: {tool.description}" for tool in self.tools.values()])
        self.system_prompt += f"\n\nYou have access to the following tools:\n{tools_description}\n"
    
    def get_tools_description(self) -> str:
        """Get a formatted description of all tools available to the agent."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool_name, tool in self.tools.items():
            descriptions.append(f"{tool_name}: {tool.description}")
        
        return "\n".join(descriptions)
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Use a specific tool from the agent's toolbox."""
        if tool_name not in self.tools:
            available_tools = ', '.join(self.tools.keys())
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
        
        return self.tools[tool_name].execute(*args, **kwargs)
    
    @abstractmethod
    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process an incoming message and return a response."""
        pass

    @abstractmethod
    async def run(self, query: str) -> str:
        """Main entry point for running the agent."""
        pass
    
    def activate(self) -> None:
        """Activate the agent."""
        self.is_active = True
        logger.info(f"Agent {self.name} ({self.id}) activated")
    
    def deactivate(self) -> None:
        """Deactivate the agent."""
        self.is_active = False
        logger.info(f"Agent {self.name} ({self.id}) deactivated")
    
    def handoff_to(self, agent_id: str, context: Dict[str, Any]) -> None:
        """Add a handoff request to the queue."""
        self.handoff_queue.append((agent_id, context))
        logger.info(f"Agent {self.name} requested handoff to agent {agent_id}")
    
    def get_next_handoff(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get the next handoff request from the queue."""
        if self.handoff_queue:
            return self.handoff_queue.pop(0)
        return None
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the agent's system prompt."""
        self.system_prompt = new_prompt


class LLMAgent(BaseAgent):
    """Agent that uses an LLM API to generate responses."""
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 model: str = "llama3-70b-8192-versatile",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 provider: str = "groq"):
        super().__init__(name=name,
                         description=description,
                         model=model,
                         temperature=temperature,
                         max_tokens=max_tokens)
        
        self.provider = provider
        self.llm_handler = LLMHandler()
    
    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message using the LLM to generate a response."""
        # Add message to conversation history
        self.memory.add_to_conversation('user', message)
        
        # Prepare context from memory if not provided
        if context is None:
            context = {}
        
        # Add conversation history to context
        context['conversation_history'] = self.memory.get_conversation_history(max_messages=10)
        
        # Prepare the messages for the API
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add relevant conversation history
        for msg in context['conversation_history']:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Only add the current message if it's not already in the history
        if not messages or messages[-1]['content'] != message:
            messages.append({"role": "user", "content": message})
        
        # Combine messages into a prompt string
        prompt_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        
        try:
            response_content = await self.llm_handler.generate(
                prompt=prompt_str,
                model=self.model,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return {"error": f"API error: {str(e)}"}
        
        # Check for tool usage in the response
        tool_request = self._extract_tool_request(response_content)
        if tool_request:
            tool_name = tool_request.get('tool_name')
            tool_args = tool_request.get('args', {})
            
            if tool_name in self.tools:
                # Execute the tool
                tool_result = self.use_tool(tool_name, **tool_args)
                
                # Add tool result to context and generate a new response
                messages.append({"role": "assistant", "content": response_content})
                messages.append({"role": "system", "content": f"Tool {tool_name} result: {json.dumps(tool_result)}"})
                
                # Create updated prompt
                updated_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
                
                # Generate a new response with the tool results
                try:
                    response_content = await self.llm_handler.generate(
                        prompt=updated_prompt,
                        model=self.model,
                        provider=self.provider,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                except Exception as e:
                    logger.error(f"LLM API error after tool use: {e}")
                    return {"error": f"API error after tool use: {str(e)}"}
        
        # Check for handoff requests in the response
        handoff_request = self._extract_handoff_request(response_content)
        if handoff_request:
            agent_id = handoff_request.get('agent_id')
            handoff_context = handoff_request.get('context', {})
            
            # Add current conversation to handoff context
            handoff_context['conversation_history'] = self.memory.get_conversation_history()
            
            # Queue the handoff
            self.handoff_to(agent_id, handoff_context)
            
            # Modify response to indicate handoff
            response_content = self._remove_handoff_directive(response_content)
            response_content += f"\n\n[Note: This conversation will be continued by another agent.]"
        
        # Add assistant response to conversation history
        self.memory.add_to_conversation('assistant', response_content)
        
        return {
            "response": response_content,
            "conversation_id": self.id,
            "handoff": bool(handoff_request)
        }
    
    async def run(self, query: str) -> str:
        """Main entry point for running the agent with a query."""
        result = await self.process(query)
        # Return just the response string for compatibility with MCP
        return result.get("response", "")
    
    def _extract_tool_request(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool usage request from response text."""
        # Look for patterns like: "[[TOOL:tool_name{args}]]"
        pattern = r"\[\[TOOL:(.*?)(\{.*?\})?\]\]"
        match = re.search(pattern, text)
        
        if match:
            tool_name = match.group(1).strip()
            args_str = match.group(2) if match.group(2) else "{}"
            
            try:
                args = json.loads(args_str)
                return {"tool_name": tool_name, "args": args}
            except json.JSONDecodeError:
                logger.error(f"Invalid tool arguments format: {args_str}")
        
        return None
    
    def _extract_handoff_request(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract handoff request from response text."""
        # Look for patterns like: "[[HANDOFF:agent_id{context}]]"
        pattern = r"\[\[HANDOFF:(.*?)(\{.*?\})?\]\]"
        match = re.search(pattern, text)
        
        if match:
            agent_id = match.group(1).strip()
            context_str = match.group(2) if match.group(2) else "{}"
            
            try:
                context = json.loads(context_str)
                return {"agent_id": agent_id, "context": context}
            except json.JSONDecodeError:
                logger.error(f"Invalid handoff context format: {context_str}")
        
        return None
    
    def _remove_handoff_directive(self, text: str) -> str:
        """Remove handoff directive from the response text."""
        pattern = r"\[\[HANDOFF:.*?\{.*?\}?\]\]"
        return re.sub(pattern, "", text).strip()


class AgentManager:
    """Manages multiple agents, handling activation, deactivation, and handoffs."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.active_agent_id: Optional[str] = None
    
    def register_agent(self, agent: BaseAgent) -> str:
        """Register an agent with the manager."""
        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.id})")
        return agent.id
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "is_active": agent.is_active
            }
            for agent in self.agents.values()
        ]
    
    def set_active_agent(self, agent_id: str) -> bool:
        """Set the active agent."""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        # Deactivate current active agent if any
        if self.active_agent_id and self.active_agent_id in self.agents:
            self.agents[self.active_agent_id].deactivate()
        
        # Activate new agent
        self.agents[agent_id].activate()
        self.active_agent_id = agent_id
        logger.info(f"Set active agent to: {self.agents[agent_id].name} ({agent_id})")
        return True
    
    async def process_message(self, message: str, agent_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message with the specified agent or the active agent."""
        # Determine which agent to use
        target_agent_id = agent_id or self.active_agent_id
        
        if not target_agent_id or target_agent_id not in self.agents:
            return {"error": "No active agent available"}
        
        agent = self.agents[target_agent_id]
        
        # Process the message
        response = await agent.process(message, context)
        
        # Check for handoffs
        handoff = agent.get_next_handoff()
        if handoff:
            next_agent_id, handoff_context = handoff
            
            if next_agent_id in self.agents:
                # Set the new agent as active
                self.set_active_agent(next_agent_id)
                
                # Add handoff information to the response
                response["handoff"] = {
                    "from_agent": agent.name,
                    "to_agent": self.agents[next_agent_id].name,
                    "to_agent_id": next_agent_id
                }
                
                # Process continuation with new agent if requested
                if handoff_context.get("auto_continue", False):
                    continuation_response = await self.agents[next_agent_id].process(
                        "Please continue the conversation based on the context provided.",
                        handoff_context
                    )
                    response["continuation"] = continuation_response
            else:
                logger.error(f"Handoff target agent {next_agent_id} not found")
                response["handoff_error"] = f"Target agent {next_agent_id} not found"
        
        return response
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the manager."""
        if agent_id not in self.agents:
            return False
        
        # If this is the active agent, clear active agent
        if self.active_agent_id == agent_id:
            self.active_agent_id = None
        
        # Remove the agent
        agent = self.agents.pop(agent_id)
        agent.deactivate()
        logger.info(f"Unregistered agent: {agent.name} ({agent.id})")
        return True


# Example specialized agent classes

class ResearcherAgent(LLMAgent):
    """An agent specialized for research tasks."""
    
    def __init__(self, name: str = "Researcher Agent", **kwargs):
        description = "I specialize in research tasks, including information gathering, analysis, and creating research proposals."
        super().__init__(name=name, description=description, **kwargs)
        
        # Add research-specific system prompt
        research_prompt = """
        As a Researcher Agent, your primary role is to help users with scientific research.
        When responding:
        1. Break down complex research topics into manageable components
        2. Cite relevant literature and methodologies 
        3. Generate hypotheses and suggest experimental designs
        4. Identify potential limitations and challenges
        5. Suggest follow-up research directions
        """
        self.update_system_prompt(self.system_prompt + research_prompt)


class CriticAgent(LLMAgent):
    """An agent specialized for critique and review."""
    
    def __init__(self, name: str = "Critic Agent", **kwargs):
        description = "I specialize in critical analysis, review, and identifying weaknesses in research proposals."
        super().__init__(name=name, description=description, **kwargs)
        
        # Add critic-specific system prompt
        critic_prompt = """
        As a Critic Agent, your primary role is to evaluate research proposals and scientific ideas.
        When responding:
        1. Identify logical fallacies and methodological weaknesses
        2. Question assumptions and highlight potential biases
        3. Suggest alternative approaches or interpretations
        4. Evaluate the potential impact and novelty
        5. Provide constructive feedback for improvement
        """
        self.update_system_prompt(self.system_prompt + critic_prompt)