"""
Agent Core Module

This module provides an improved base agent with:
- Robust state management
- Event system for agent activity monitoring
- Enhanced error handling and recovery
- Better conversation history management
"""

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import json
import traceback

from backend.app.utils.llm import LLMHandler
from backend.app.utils.guardrails import Guardrails, GuardrailsResult
from backend.app.utils.memory import MemoryManager
from backend.app.utils.errors import (
    AgentError, 
    LLMError, 
    RateLimitError, 
    ContextLengthError,
    GuardrailsError
)

# Configure logging
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """States an agent can be in during its lifecycle."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"

class MessageRole(Enum):
    """Roles for conversation messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

@dataclass
class Message:
    """Structured representation of a conversation message."""
    role: MessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentEvent:
    """Event emitted by the agent during operation."""
    type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None

class BaseAgent:
    """
    Enhanced base agent with improved state management and error handling.
    
    This class provides the foundation for all agent types with common
    functionality for state management, conversation history, event handling,
    and error recovery.
    """
    
    def __init__(
        self,
        llm_handler: LLMHandler,
        guardrails: Guardrails,
        memory_manager: Optional[MemoryManager] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            llm_handler: Handler for LLM interactions
            guardrails: Content safety system
            memory_manager: Optional memory management system
            config: Agent configuration
            agent_id: Optional unique ID for this agent instance
        """
        self.llm_handler = llm_handler
        self.guardrails = guardrails
        self.memory_manager = memory_manager or MemoryManager()
        self.config = config or {}
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = AgentState.INITIALIZING
        
        # Initialize conversation history
        self.conversation_history: List[Message] = []
        
        # Initialize event handlers
        self.event_handlers: Dict[str, List[Callable[[AgentEvent], None]]] = {}
        
        # Add system prompt if provided
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            self.add_message(MessageRole.SYSTEM, system_prompt)
        
        # Set state to ready
        self.state = AgentState.READY
        self._emit_event("agent_initialized", {
            "agent_type": self.__class__.__name__,
            "config": {k: v for k, v in self.config.items() if k != "api_key"}
        })
    
    def add_message(
        self, 
        role: Union[MessageRole, str], 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender
            content: Message content
            metadata: Optional metadata for the message
            
        Returns:
            The created message object
        """
        # Convert string role to enum if needed
        if isinstance(role, str):
            try:
                role = MessageRole[role.upper()]
            except KeyError:
                role = MessageRole.USER
        
        # Create message
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        # Add to history
        self.conversation_history.append(message)
        
        # Emit event
        self._emit_event("message_added", {
            "message_id": message.id,
            "role": message.role.value,
            "content_length": len(content)
        })
        
        return message
    
    def process_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: Input text from the user
            context: Optional context information
            
        Returns:
            Agent's response text
            
        Raises:
            AgentError: If processing fails
        """
        # Update state
        previous_state = self.state
        self.state = AgentState.PROCESSING
        
        try:
            # Check input with guardrails
            guardrails_result = self.guardrails.check_content(user_input, context)
            if not guardrails_result.passed:
                return self._handle_guardrails_violation(guardrails_result)
            
            # Add user message to history
            self.add_message(MessageRole.USER, user_input, metadata=context)
            
            # Generate response
            response = self._generate_response(user_input, context)
            
            # Check response with guardrails
            response_check = self.guardrails.check_content(response)
            if not response_check.passed:
                response = response_check.modified_content or "I apologize, but I cannot provide that information."
            
            # Add assistant message to history
            self.add_message(MessageRole.ASSISTANT, response)
            
            # Update memory if needed
            if self.memory_manager:
                self.memory_manager.add_interaction(user_input, response, context)
            
            # Return to ready state
            self.state = AgentState.READY
            return response
            
        except Exception as e:
            # Handle error and return to previous state if appropriate
            error_response = self._handle_error(e)
            self.state = previous_state if previous_state != AgentState.ERROR else AgentState.READY
            return error_response
    
    def _generate_response(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response to the user input.
        
        Args:
            user_input: Input text from the user
            context: Optional context information
            
        Returns:
            Generated response text
            
        Raises:
            LLMError: If LLM request fails
        """
        # Prepare messages for LLM
        messages = self._prepare_messages_for_llm()
        
        # Get LLM parameters from config
        params = {
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens"),
            "top_p": self.config.get("top_p", 1.0),
        }
        
        # Generate response with retry logic
        max_retries = self.config.get("max_retries", 3)
        retry_delay = self.config.get("retry_delay", 1.0)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.llm_handler.generate_text(messages, **params)
                elapsed_time = time.time() - start_time
                
                self._emit_event("llm_response_received", {
                    "elapsed_time": elapsed_time,
                    "attempt": attempt + 1,
                    "tokens": len(response) // 4  # Rough estimate
                })
                
                return response
                
            except RateLimitError as e:
                # Special handling for rate limits
                if attempt < max_retries - 1:
                    self._emit_event("rate_limit_retry", {
                        "attempt": attempt + 1,
                        "delay": retry_delay,
                        "error": str(e)
                    })
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
                    
            except ContextLengthError as e:
                # Handle context length issues by summarizing history
                self._emit_event("context_length_exceeded", {
                    "current_length": len(str(messages)),
                    "error": str(e)
                })
                
                self._summarize_conversation_history()
                messages = self._prepare_messages_for_llm()  # Refresh messages after summarization
                
            except Exception as e:
                # Log other errors and retry if possible
                if attempt < max_retries - 1:
                    self._emit_event("llm_error_retry", {
                        "attempt": attempt + 1,
                        "error": str(e)
                    })
                    time.sleep(retry_delay)
                else:
                    raise LLMError(f"Failed to generate response after {max_retries} attempts: {e}")
        
        # Should not reach here, but just in case
        raise LLMError("Failed to generate response")
    
    def _prepare_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Prepare conversation history for LLM API format.
        
        Returns:
            List of message dictionaries in LLM-compatible format
        """
        # Start with recent messages, limited by max_context_length
        max_messages = self.config.get("max_context_messages", 10)
        
        # Always include system messages
        system_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.conversation_history
            if msg.role == MessageRole.SYSTEM
        ]
        
        # Get recent non-system messages
        recent_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.conversation_history[-max_messages:]
            if msg.role != MessageRole.SYSTEM
        ]
        
        # Combine system and recent messages
        return system_messages + recent_messages
    
    def _summarize_conversation_history(self) -> None:
        """
        Summarize conversation history to reduce context length.
        """
        # Skip if not enough messages to summarize
        if len(self.conversation_history) < 4:
            return
        
        # Keep system messages and last 2 exchanges (4 messages)
        system_messages = [msg for msg in self.conversation_history if msg.role == MessageRole.SYSTEM]
        recent_messages = self.conversation_history[-4:]
        
        # Summarize the rest
        messages_to_summarize = [
            msg for msg in self.conversation_history 
            if msg.role != MessageRole.SYSTEM and msg not in recent_messages
        ]
        
        if not messages_to_summarize:
            return
        
        # Create summary prompt
        summary_prompt = "Summarize the following conversation concisely:\n\n"
        for msg in messages_to_summarize:
            summary_prompt += f"{msg.role.value.capitalize()}: {msg.content}\n\n"
        
        try:
            # Generate summary
            summary = self.llm_handler.generate_text([
                {"role": "user", "content": summary_prompt}
            ], temperature=0.3, max_tokens=200)
            
            # Create summary message
            summary_message = Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {summary}",
                metadata={"summary": True, "original_messages": len(messages_to_summarize)}
            )
            
            # Replace old messages with summary
            self.conversation_history = system_messages + [summary_message] + recent_messages
            
            self._emit_event("conversation_summarized", {
                "original_message_count": len(messages_to_summarize),
                "new_message_count": len(self.conversation_history)
            })
            
        except Exception as e:
            logger.warning(f"Failed to summarize conversation: {e}")
            # Fall back to simple truncation
            self.conversation_history = system_messages + recent_messages
    
    def _handle_guardrails_violation(self, result: GuardrailsResult) -> str:
        """
        Handle content that violates guardrails.
        
        Args:
            result: Guardrails check result
            
        Returns:
            Response to the user
        """
        violation_type = result.violations[0].type.value if result.violations else "unknown"
        
        self._emit_event("guardrails_violation", {
            "violation_type": violation_type,
            "severity": result.highest_severity,
            "risk_score": result.risk_score
        })
        
        # Update state to reflect error
        self.state = AgentState.ERROR
        
        # Get appropriate response based on violation type
        responses = {
            "harmful_content": "I cannot provide information on harmful content.",
            "personally_identifiable_information": "I cannot process personal information.",
            "profanity": "I'd appreciate if we could keep our conversation respectful.",
            "prohibited_topic": "I'm not able to discuss this topic.",
            "security_risk": "This request poses a security risk and cannot be processed.",
            "prompt_injection": "I've detected an attempt to override my guidelines."
        }
        
        return responses.get(violation_type, "I cannot process that request due to content safety restrictions.")
    
    def _handle_error(self, error: Exception) -> str:
        """
        Handle errors during processing.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Error response to the user
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log error details
        logger.error(
            f"Agent error: {error_type}: {error_message}",
            exc_info=True
        )
        
        # Update state and emit event
        self.state = AgentState.ERROR
        self._emit_event("agent_error", {
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback.format_exc()
        })
        
        # Determine user-facing error message
        if isinstance(error, RateLimitError):
            return "I'm currently experiencing high demand. Please try again in a moment."
        elif isinstance(error, ContextLengthError):
            return "Our conversation has grown too long. Let's start a new topic."
        elif isinstance(error, LLMError):
            return "I'm having trouble generating a response. Let's try something else."
        elif isinstance(error, GuardrailsError):
            return "I cannot process that request due to content safety restrictions."
        else:
            return "I encountered an unexpected issue. Could you try again or rephrase your request?"
    
    def register_event_handler(self, event_type: str, handler: Callable[[AgentEvent], None]) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = AgentEvent(type=event_type, data=data, agent_id=self.agent_id)
        
        # Call handlers for this event type
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
        
        # Call handlers for "all" events
        for handler in self.event_handlers.get("all", []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in 'all' event handler: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary with agent state information
        """
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "message_count": len(self.conversation_history),
            "last_message_time": self.conversation_history[-1].timestamp if self.conversation_history else None,
            "memory_size": self.memory_manager.size if self.memory_manager else 0,
            "config": {k: v for k, v in self.config.items() if k not in ["api_key", "system_prompt"]}
        }
    
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        
        This preserves system messages but clears conversation history.
        """
        # Keep system messages
        system_messages = [msg for msg in self.conversation_history if msg.role == MessageRole.SYSTEM]
        
        # Clear history and add back system messages
        self.conversation_history = system_messages.copy()
        
        # Reset memory
        if self.memory_manager:
            self.memory_manager.clear()
        
        # Reset state
        self.state = AgentState.READY
        
        self._emit_event("agent_reset", {
            "retained_messages": len(system_messages)
        })
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the agent.
        
        This should be called when the agent is no longer needed.
        """
        # Update state
        self.state = AgentState.TERMINATED
        
        # Clear references to external resources
        self.llm_handler = None
        self.guardrails = None
        
        # Clear memory
        if self.memory_manager:
            self.memory_manager.clear()
            self.memory_manager = None
        
        # Clear conversation history
        self.conversation_history = []
        
        # Clear event handlers
        self.event_handlers = {}
        
        self._emit_event("agent_terminated", {})