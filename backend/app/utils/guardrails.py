# backend/app/utils/guardrails.py

"""
Guardrails Checker Module

Provides basic input validation and output sanitization to ensure
user queries and model responses stay within allowed domains and
content safety guidelines.
"""

import re
from typing import Dict


class GuardrailsChecker:
    """
    Simple content safety and domain guardrails.
    - Ensures topics are in Computer Science or Data Science.
    - Blocks disallowed words/patterns.
    - Sanitizes model output if needed.
    """

    def __init__(self):
        # Allowed domain keywords for initial query filtering
        self.allowed_domains = ["computer science", "data science"]
        # Example blacklist of disallowed terms (extend as needed)
        self.blacklist = [
            r"\bterrorism\b",
            r"\bself[-\s]?harm\b",
            r"\bpolitics?\b",
        ]
        # Compile blacklist regexes
        self._blacklist_patterns = [re.compile(pat, re.IGNORECASE) for pat in self.blacklist]

    def check_input(self, text: str) -> Dict[str, str]:
        """
        Validate user input.

        Returns a dict:
          - passed: bool
          - reason: str (empty if passed)
          - message: str (error message if not passed)
        """
        lower = text.lower()

        # Domain filter
        if not any(domain in lower for domain in self.allowed_domains):
            return {
                "passed": False,
                "reason": "domain_filter",
                "message": "Please ask about a Computer Science or Data Science topic."
            }

        # Blacklist filter
        for pattern in self._blacklist_patterns:
            if pattern.search(text):
                return {
                    "passed": False,
                    "reason": "content_safety",
                    "message": "Your query contains disallowed content."
                }

        return {"passed": True, "reason": "", "message": ""}

    def sanitize_output(self, text: str) -> str:
        """
        Sanitize model output.

        - Removes any blacklisted content.
        - (Optional) Trims extremely long responses.
        """
        # Remove any blacklisted terms from output
        for pattern in self._blacklist_patterns:
            text = pattern.sub("[REDACTED]", text)

        # Optionally trim overly long outputs
        max_chars = 20000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Response truncated]"

        return text