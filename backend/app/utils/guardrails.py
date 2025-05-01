# backend/app/utils/guardrails.py

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Load the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class GuardrailViolationType(str, Enum):
    """Types of guardrail violations"""
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    CODE_EXECUTION = "code_execution"
    SENSITIVE_DATA = "sensitive_data"
    SYSTEM_INFO_LEAKAGE = "system_info_leakage"
    UNSAFE_CODE = "unsafe_code"
    FORMATTING = "formatting"
    MAX_LENGTH = "max_length"
    CUSTOM = "custom"


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation"""
    type: GuardrailViolationType
    rule_name: str
    message: str
    severity: str = "high"  # high, medium, low
    matched_content: Optional[str] = None


@dataclass
class GuardrailResult:
    """Result of a guardrail check"""
    passed: bool
    violations: List[GuardrailViolation] = None
    sanitized_content: Optional[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class GuardRailsConfig:
    """Configuration for guardrails"""
    
    def __init__(self, config_path: str = "backend/app/guardrails/guardrails.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load guardrails configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load guardrails config: {str(e)}")
            # Return default minimal config
            return {
                "input": {
                    "type": "string",
                    "rules": []
                },
                "output": {
                    "type": "string",
                    "rules": []
                }
            }
    
    def reload(self):
        """Reload configuration from disk"""
        self.config = self._load_config()


class GuardRailsChecker:
    """Enhanced guardrails implementation with detailed violation reporting"""
    
    def __init__(self, config_path: str = "backend/app/guardrails/guardrails.json"):
        self.config = GuardRailsConfig(config_path)
        self.client = client
        
        # Map rule names to violation types
        self.rule_type_mapping = {
            "max_length": GuardrailViolationType.MAX_LENGTH,
            "no_pii": GuardrailViolationType.PII,
            "no_prompt_injection": GuardrailViolationType.PROMPT_INJECTION,
            "no_code_execution": GuardrailViolationType.CODE_EXECUTION,
            "no_sensitive_output": GuardrailViolationType.SENSITIVE_DATA,
            "no_system_info_leakage": GuardrailViolationType.SYSTEM_INFO_LEAKAGE,
            "no_code_output": GuardrailViolationType.UNSAFE_CODE,
            "structured_output_required": GuardrailViolationType.FORMATTING
        }
    
    def check_input(self, user_input: str) -> GuardrailResult:
        """
        Validate and sanitize user input against guardrails
        
        Args:
            user_input: The user input to check
            
        Returns:
            GuardrailResult object with validation details
        """
        if not user_input:
            return GuardrailResult(passed=True, sanitized_content="")
            
        input_rules = self.config.config.get("input", {}).get("rules", [])
        violations = []
        
        for rule in input_rules:
            rule_name = rule.get("name", "")
            rule_type = rule.get("type", "")
            
            # Apply different types of checks based on rule type
            if rule_type == "length" and len(user_input) > rule.get("max", 10000):
                violation_type = GuardrailViolationType.MAX_LENGTH
                violations.append(GuardrailViolation(
                    type=violation_type,
                    rule_name=rule_name,
                    message=rule.get("error_message", "Input too long")
                ))
            
            elif rule_type == "regex":
                import re
                pattern = rule.get("pattern", "")
                try:
                    match = re.search(pattern, user_input)
                    if match:
                        violation_type = self.rule_type_mapping.get(
                            rule_name, GuardrailViolationType.CUSTOM
                        )
                        violations.append(GuardrailViolation(
                            type=violation_type,
                            rule_name=rule_name,
                            message=rule.get("error_message", "Pattern matched"),
                            matched_content=match.group(0)
                        ))
                except Exception as e:
                    logger.error(f"Regex error in rule {rule_name}: {str(e)}")
        
        # Determine result
        if violations:
            # Format a user-friendly message
            message = self._format_violation_message(violations)
            return GuardrailResult(
                passed=False, 
                violations=violations,
                sanitized_content=message
            )
        
        return GuardrailResult(passed=True, sanitized_content=user_input)
    
    def sanitize_output(self, model_output: str) -> str:
        """
        Sanitize and validate model output
        
        Args:
            model_output: The model output to check
            
        Returns:
            Sanitized output or warning message
        """
        if not model_output:
            return ""
            
        result = self.check_output(model_output)
        
        if not result.passed:
            # Format a user-friendly message
            sanitized = self._sanitize_content(model_output, result.violations)
            return sanitized
        
        return model_output
    
    def check_output(self, model_output: str) -> GuardrailResult:
        """
        Check model output against guardrails
        
        Args:
            model_output: The model output to check
            
        Returns:
            GuardrailResult object with validation details
        """
        if not model_output:
            return GuardrailResult(passed=True, sanitized_content="")
            
        output_rules = self.config.config.get("output", {}).get("rules", [])
        violations = []
        
        for rule in output_rules:
            rule_name = rule.get("name", "")
            rule_type = rule.get("type", "")
            
            if rule_type == "regex":
                import re
                pattern = rule.get("pattern", "")
                try:
                    match = re.search(pattern, model_output)
                    if match:
                        violation_type = self.rule_type_mapping.get(
                            rule_name, GuardrailViolationType.CUSTOM
                        )
                        violations.append(GuardrailViolation(
                            type=violation_type,
                            rule_name=rule_name,
                            message=rule.get("error_message", "Pattern matched"),
                            matched_content=match.group(0)
                        ))
                except Exception as e:
                    logger.error(f"Regex error in rule {rule_name}: {str(e)}")
        
        # Determine result
        if violations:
            return GuardrailResult(
                passed=False, 
                violations=violations
            )
        
        return GuardrailResult(passed=True, sanitized_content=model_output)
    
    def _format_violation_message(self, violations: List[GuardrailViolation]) -> str:
        """Format a user-friendly message from violations"""
        if not violations:
            return ""
            
        if len(violations) == 1:
            return f"⚠️ {violations[0].message}"
            
        messages = [f"⚠️ Content policy violation:"]
        for i, v in enumerate(violations, 1):
            messages.append(f"{i}. {v.message}")
        
        return "\n".join(messages)
    
    def _sanitize_content(self, content: str, violations: List[GuardrailViolation]) -> str:
        """
        Attempt to sanitize content by removing violating patterns
        
        This is a basic implementation - in a real system, you might want to use
        a more sophisticated approach or a content moderation API
        """
        if not violations:
            return content
            
        sanitized = content
        for violation in violations:
            if violation.matched_content:
                # Replace the matched content with asterisks
                sanitized = sanitized.replace(
                    violation.matched_content, 
                    "*" * len(violation.matched_content)
                )
        
        # Add a warning header
        return f"[⚠️ Some content has been sanitized]\n\n{sanitized}"
    
    async def guarded_chat(self, user_input: str, chat_func) -> Dict[str, Any]:
        """
        Run a chat through input/output validation
        
        Args:
            user_input: The user input
            chat_func: Async function to call the model
            
        Returns:
            Dict with trace and final output
        """
        # Check input
        input_check = self.check_input(user_input)
        if not input_check.passed:
            return {
                "trace": [{
                    "stage": "guardrails", 
                    "result": "blocked", 
                    "violations": [v.__dict__ for v in input_check.violations]
                }],
                "final_output": input_check.sanitized_content
            }
            
        # Process with the model
        try:
            model_response = await chat_func(user_input)
        except Exception as e:
            logger.error(f"Chat function error: {str(e)}")
            return {
                "trace": [{"stage": "model", "result": "error", "error": str(e)}],
                "final_output": f"An error occurred while processing your request: {str(e)}"
            }
            
        # Check output
        output_check = self.check_output(model_response)
        if not output_check.passed:
            sanitized = self._sanitize_content(model_response, output_check.violations)
            return {
                "trace": [{
                    "stage": "guardrails", 
                    "result": "sanitized", 
                    "violations": [v.__dict__ for v in output_check.violations]
                }],
                "final_output": sanitized
            }
            
        # All good
        return {
            "trace": [{"stage": "guardrails", "result": "passed"}],
            "final_output": model_response
        }