"""
Error handling utilities for the AI research lab agents.
This module provides custom exception classes for better error handling and reporting.
"""

class AgentError(Exception):
    """Base exception class for all agent-related errors."""
    
    def __init__(self, message="An error occurred in the agent system", agent_id=None):
        self.agent_id = agent_id
        self.message = f"Agent {agent_id}: {message}" if agent_id else message
        super().__init__(self.message)


class AgentInitializationError(AgentError):
    """Exception raised during agent initialization."""
    
    def __init__(self, message="Failed to initialize agent", agent_id=None, details=None):
        self.details = details
        super_message = f"{message}. Details: {details}" if details else message
        super().__init__(super_message, agent_id)


class AgentCommunicationError(AgentError):
    """Exception raised when there's an error in agent communication."""
    
    def __init__(self, message="Communication error", agent_id=None, target_id=None):
        self.target_id = target_id
        super_message = f"{message} with target {target_id}" if target_id else message
        super().__init__(super_message, agent_id)


class LLMError(AgentError):
    """Exception raised when there's an error with the LLM provider."""
    
    def __init__(self, message="Error with LLM provider", agent_id=None, provider=None):
        self.provider = provider
        super_message = f"{message} ({provider})" if provider else message
        super().__init__(super_message, agent_id)


class GuardrailViolationError(AgentError):
    """Exception raised when content violates guardrails."""
    
    def __init__(self, message="Content violates guardrails", agent_id=None, violation_type=None):
        self.violation_type = violation_type
        super_message = f"{message}. Type: {violation_type}" if violation_type else message
        super().__init__(super_message, agent_id)


class MemoryError(AgentError):
    """Exception raised when there's an error with the agent's memory system."""
    
    def __init__(self, message="Error in memory system", agent_id=None, operation=None):
        self.operation = operation
        super_message = f"{message} during {operation}" if operation else message
        super().__init__(super_message, agent_id)