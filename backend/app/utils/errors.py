# backend/app/utils/errors.py

"""
Core exception definitions for the AI Research Lab.
Provides simple, descriptive errors for agents and infrastructure.
"""

class AgentError(Exception):
    """Base exception for all agent-related errors."""
    def __init__(self, message: str, agent_id: str = None):
        prefix = f"[Agent: {agent_id}] " if agent_id else ""
        super().__init__(f"{prefix}{message}")


class GuardrailViolationError(AgentError):
    """Raised when input or output violates guardrails."""
    def __init__(self, message: str = "Content violates guardrails", agent_id: str = None):
        super().__init__(message, agent_id)
        self.violation_type = message  # you can use this for logging/metrics


class LLMError(AgentError):
    """Raised when an LLM call fails or returns an error."""
    def __init__(self, message: str = "LLM provider error", agent_id: str = None):
        super().__init__(message, agent_id)


class MemoryError(AgentError):
    """Raised on session or memory-store failures."""
    def __init__(self, message: str = "Memory operation failed", agent_id: str = None):
        super().__init__(message, agent_id)


class CommunicationError(AgentError):
    """Raised on errors between agents, or client-server transport failures."""
    def __init__(self, message: str = "Communication failure", agent_id: str = None):
        super().__init__(message, agent_id)