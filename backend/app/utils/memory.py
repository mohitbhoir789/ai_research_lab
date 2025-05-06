"""
Memory management utilities for AI research lab agents.
Provides functionality for storing, retrieving, and managing agent memory and conversation history.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationHistory:
    """Manages conversation history for agents."""
    
    def __init__(self, max_history_length: int = 100):
        self.messages: List[Dict[str, Any]] = []
        self.max_history_length = max_history_length
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history."""
        timestamp = datetime.now().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        if metadata:
            message["metadata"] = metadata
            
        self.messages.append(message)
        
        # Trim history if it exceeds the maximum length
        if len(self.messages) > self.max_history_length:
            self.messages = self.messages[-self.max_history_length:]
            logger.info(f"Trimmed conversation history to {self.max_history_length} messages")
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history, optionally limited to the most recent messages."""
        if limit is None or limit >= len(self.messages):
            return self.messages
        return self.messages[-limit:]
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        logger.info("Conversation history cleared")


class AgentMemory:
    """Manages persistent memory for agents, including conversation history and state."""
    
    def __init__(self, agent_id: str, max_history_length: int = 100):
        self.agent_id = agent_id
        self.conversation = ConversationHistory(max_history_length)
        self.state: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "interactions": 0
        }
    
    def update_state(self, key: str, value: Any) -> None:
        """Update a value in the agent's state."""
        self.state[key] = value
        logger.debug(f"Agent {self.agent_id} state updated: {key}={value}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the agent's state."""
        return self.state.get(key, default)
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message to the conversation history."""
        self.conversation.add_message("user", content, metadata)
        self.metadata["interactions"] += 1
    
    def add_agent_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an agent message to the conversation history."""
        self.conversation.add_message("agent", content, metadata)
    
    def get_formatted_history(self, limit: Optional[int] = None, max_chars: int = 1000) -> List[Dict[str, str]]:
        """
        Get formatted conversation history suitable for LLM context, limited by character count.
        
        Args:
            limit: Optional message count limit
            max_chars: Maximum total characters allowed (default: 1000)
            
        Returns:
            List of formatted messages, potentially truncated to respect character limit
        """
        messages = self.conversation.get_history(limit)
        formatted_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        
        # Calculate total characters
        total_chars = sum(len(msg["content"]) for msg in formatted_messages)
        
        # If under limit, return as is
        if total_chars <= max_chars:
            return formatted_messages
            
        # Otherwise need to trim messages to fit under max_chars
        result = []
        char_count = 0
        
        # Start from most recent messages (reversed order)
        for msg in reversed(formatted_messages):
            content_len = len(msg["content"])
            
            # If adding this message would exceed limit, truncate it
            if char_count + content_len > max_chars:
                # Calculate remaining space
                remaining = max_chars - char_count
                if remaining > 50:  # Only add if we can include meaningful content
                    truncated_content = msg["content"][:remaining-3] + "..."
                    result.insert(0, {"role": msg["role"], "content": truncated_content})
                break
            
            # Otherwise add the whole message
            char_count += content_len
            result.insert(0, msg)
            
            # Stop if we're at the limit
            if char_count >= max_chars:
                break
                
        logger.debug(f"History truncated from {total_chars} to {char_count} characters")
        return result
    
    def summarize_memory(self) -> Dict[str, Any]:
        """Summarize the agent's memory for logging or debugging."""
        return {
            "agent_id": self.agent_id,
            "state_keys": list(self.state.keys()),
            "message_count": len(self.conversation.messages),
            "metadata": self.metadata
        }
    
    def save(self) -> Dict[str, Any]:
        """Prepare memory contents for persistence."""
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "conversation": self.conversation.messages,
            "metadata": self.metadata
        }
    
    @classmethod
    def load(cls, data: Dict[str, Any]) -> 'AgentMemory':
        """Create an AgentMemory instance from saved data."""
        memory = cls(data["agent_id"])
        memory.state = data.get("state", {})
        memory.metadata = data.get("metadata", {})
        memory.conversation.messages = data.get("conversation", [])
        return memory