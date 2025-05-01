"""
Enhanced Error Handling System

This module provides a comprehensive error handling system with:
- Specialized error types for different failure scenarios
- Detailed error reporting
- Recovery mechanisms
- Consistent logging patterns
"""

import logging
import traceback
import json
from typing import Dict, Any, Optional, List, Union
import time
from enum import Enum
from dataclasses import dataclass, field
import sys
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"  # System cannot continue, requires immediate attention
    ERROR = "error"        # Operation failed, but system can continue
    WARNING = "warning"    # Potential issue that didn't cause failure
    INFO = "info"          # Informational error, minimal impact

@dataclass
class ErrorContext:
    """Context information for an error."""
    component: str
    operation: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    additional: Dict[str, Any] = field(default_factory=dict)

class BaseError(Exception):
    """
    Base class for all application errors.
    
    Provides consistent error handling patterns and detailed context.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
        recoverable: bool = True,
        retry_after: Optional[float] = None,
    ):
        """
        Initialize the error.
        
        Args:
            message: Human-readable error message
            severity: Severity level of the error
            context: Error context information
            original_error: Original exception if this is a wrapper
            error_code: Machine-readable error code
            recoverable: Whether the error is potentially recoverable
            retry_after: Suggested time to wait before retry (seconds)
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context
        self.original_error = original_error
        self.error_code = error_code or self._generate_error_code()
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = time.time()
        
        # Log the error when it's created
        self._log_error()
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code."""
        error_type = self.__class__.__name__
        timestamp = int(time.time())
        unique_part = str(uuid.uuid4())[:8]
        return f"{error_type}-{timestamp}-{unique_part}"
    
    def _log_error(self) -> None:
        """Log the error with appropriate level and context."""
        log_method = {
            ErrorSeverity.CRITICAL: logger.critical,
            ErrorSeverity.ERROR: logger.error,
            ErrorSeverity.WARNING: logger.warning,
            ErrorSeverity.INFO: logger.info,
        }.get(self.severity, logger.error)
        
        # Format context for logging
        context_str = "No context"
        if self.context:
            try:
                safe_context = self._get_safe_context_dict()
                context_str = json.dumps(safe_context)
            except Exception as e:
                context_str = f"[Context serialization failed: {e}]"
        
        # Log with original exception traceback if available
        if self.original_error:
            log_method(
                f"{self.message} [code={self.error_code}] | Context: {context_str}",
                exc_info=self.original_error
            )
        else:
            log_method(
                f"{self.message} [code={self.error_code}] | Context: {context_str}",
                exc_info=True
            )
    
    def _get_safe_context_dict(self) -> Dict[str, Any]:
        """Get a sanitized version of the context for logging."""
        if not self.context:
            return {}
        
        # Create safe copy of context without sensitive data
        safe_context = {
            "component": self.context.component,
            "operation": self.context.operation,
            "timestamp": self.context.timestamp,
        }
        
        # Only include non-sensitive fields
        if self.context.request_id:
            safe_context["request_id"] = self.context.request_id
            
        if self.context.user_id:
            safe_context["user_id"] = self.context.user_id
        
        # Filter sensitive data from inputs
        if self.context.inputs:
            safe_inputs = {}
            sensitive_keys = ["password", "token", "api_key", "secret", "credential"]
            
            for key, value in self.context.inputs.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    safe_inputs[key] = "[REDACTED]"
                else:
                    safe_inputs[key] = value
            
            safe_context["inputs"] = safe_inputs
        
        # Include additional context data after filtering
        if self.context.additional:
            safe_additional = {
                k: v for k, v in self.context.additional.items()
                if not any(sensitive in k.lower() for sensitive in ["password", "token", "api_key"])
            }
            safe_context["additional"] = safe_additional
        
        return safe_context
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to a dictionary for API responses.
        
        Returns:
            Error information as a dictionary
        """
        result = {
            "error": {
                "type": self.__class__.__name__,
                "message": self.message,
                "code": self.error_code,
                "severity": self.severity.value,
                "timestamp": self.timestamp,
                "recoverable": self.recoverable,
            }
        }
        
        if self.retry_after is not None:
            result["error"]["retry_after"] = self.retry_after
        
        return result
    
    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.__class__.__name__}[{self.error_code}]: {self.message}"

# Component-specific errors

class ConfigError(BaseError):
    """Error related to configuration issues."""
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        missing_keys: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_file: Path to the problematic config file
            missing_keys: List of missing configuration keys
            **kwargs: Additional arguments for BaseError
        """
        context = kwargs.pop("context", None) or ErrorContext(
            component="config",
            operation="load_config",
            additional={"config_file": config_file, "missing_keys": missing_keys}
        )
        super().__init__(message, context=context, **kwargs)
        self.config_file = config_file
        self.missing_keys = missing_keys or []

class DatabaseError(BaseError):
    """Error related to database operations."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        collection: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize database error.
        
        Args:
            message: Error message
            operation: Database operation that failed
            collection: Database collection being accessed
            query: Query that caused the error
            **kwargs: Additional arguments for BaseError
        """
        context = kwargs.pop("context", None) or ErrorContext(
            component="database",
            operation=operation,
            additional={"collection": collection}
        )
        
        # Sanitize query to avoid logging sensitive data
        if query and context and hasattr(context, "additional"):
            safe_query = {
                k: (v if k not in ["password", "token", "api_key"] else "[REDACTED]")
                for k, v in query.items()
            }
            context.additional["query"] = safe_query
        
        super().__init__(message, context=context, **kwargs)
        self.operation = operation
        self.collection = collection
        self.query = query

class LLMError(BaseError):
    """Error related to Language Model operations."""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM error.
        
        Args:
            message: Error message
            model: LLM model being used
            provider: LLM provider (e.g., OpenAI, Anthropic)
            **kwargs: Additional arguments for BaseError
        """
        context = kwargs.pop("context", None) or ErrorContext(
            component="llm",
            operation="generate",
            additional={"model": model, "provider": provider}
        )
        super().__init__(message, context=context, **kwargs)
        self.model = model
        self.provider = provider

class RateLimitError(LLMError):
    """Error due to rate limiting by the LLM provider."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = 60.0,
        **kwargs
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            **kwargs: Additional arguments for LLMError
        """
        super().__init__(
            message, 
            retry_after=retry_after,
            recoverable=True,
            **kwargs
        )

class ContextLengthError(LLMError):
    """Error due to context length limits of the LLM."""
    
    def __init__(
        self,
        message: str,
        current_length: Optional[int] = None,
        max_length: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize context length error.
        
        Args:
            message: Error message
            current_length: Current context length
            max_length: Maximum allowed length
            **kwargs: Additional arguments for LLMError
        """
        context = kwargs.pop("context", None)
        if context and hasattr(context, "additional"):
            context.additional.update({
                "current_length": current_length,
                "max_length": max_length,
                "excess": current_length - max_length if current_length and max_length else None
            })
        
        super().__init__(message, **kwargs)
        self.current_length = current_length
        self.max_length = max_length

class GuardrailsError(BaseError):
    """Error related to content safety guardrails."""