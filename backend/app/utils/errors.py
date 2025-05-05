# backend/app/utils/errors.py

"""
Error Handling Module
Custom exceptions and error handling utilities for the AI Research Lab.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ResearchLabError(Exception):
    """Base exception class for AI Research Lab."""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class LLMError(ResearchLabError):
    """Exceptions related to LLM operations."""
    
    def __init__(self, message: str, provider: str, details: Optional[Dict[str, Any]] = None):
        error_code = f"LLM_{provider.upper()}_ERROR"
        super().__init__(message, error_code, details)
        self.provider = provider

class EmbeddingError(ResearchLabError):
    """Exceptions related to embedding operations."""
    
    def __init__(self, message: str, provider: str, details: Optional[Dict[str, Any]] = None):
        error_code = f"EMBEDDING_{provider.upper()}_ERROR"
        super().__init__(message, error_code, details)
        self.provider = provider

class VectorDBError(ResearchLabError):
    """Exceptions related to vector database operations."""
    
    def __init__(self, message: str, operation: str, details: Optional[Dict[str, Any]] = None):
        error_code = f"VECTORDB_{operation.upper()}_ERROR"
        super().__init__(message, error_code, details)
        self.operation = operation

class GuardrailsError(ResearchLabError):
    """Exceptions related to content guardrails."""
    
    def __init__(self, message: str, check_type: str, details: Optional[Dict[str, Any]] = None):
        error_code = f"GUARDRAILS_{check_type.upper()}_ERROR"
        super().__init__(message, error_code, details)
        self.check_type = check_type

class AgentError(ResearchLabError):
    """Exceptions related to agent operations."""
    
    def __init__(self, message: str, agent_type: str, details: Optional[Dict[str, Any]] = None):
        error_code = f"AGENT_{agent_type.upper()}_ERROR"
        super().__init__(message, error_code, details)
        self.agent_type = agent_type

class SessionError(ResearchLabError):
    """Exceptions related to session management."""
    
    def __init__(self, message: str, operation: str, details: Optional[Dict[str, Any]] = None):
        error_code = f"SESSION_{operation.upper()}_ERROR"
        super().__init__(message, error_code, details)
        self.operation = operation

def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format an exception into a standardized error response.
    
    Args:
        error: The exception to format
        
    Returns:
        Formatted error dictionary
    """
    if isinstance(error, ResearchLabError):
        response = {
            "success": False,
            "error_code": error.error_code,
            "message": error.message,
            "details": error.details
        }
        
        # Add provider-specific info for relevant errors
        if isinstance(error, (LLMError, EmbeddingError)):
            response["provider"] = error.provider
        elif isinstance(error, AgentError):
            response["agent_type"] = error.agent_type
        
        return response
    else:
        # Handle unknown errors
        return {
            "success": False,
            "error_code": "UNKNOWN_ERROR",
            "message": str(error),
            "details": {"type": type(error).__name__}
        }

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with additional context.
    
    Args:
        error: The exception to log
        context: Optional additional context
    """
    context = context or {}
    
    if isinstance(error, ResearchLabError):
        logger.error(
            f"{error.error_code}: {error.message}",
            extra={
                "error_code": error.error_code,
                "details": error.details,
                **context
            }
        )
    else:
        logger.error(
            f"Unknown error: {str(error)}",
            extra={
                "error_type": type(error).__name__,
                **context
            },
            exc_info=True
        )