# backend/app/utils/guardrails.py

import os
from openai import OpenAI
from guardrails import Guard, openai
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# Load the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load guardrail configuration
guard = Guard.from_rail("backend/app/guardrails/guardrails.json")

class GuardRailsChecker:
    def __init__(self):
        self.guard = guard
        self.client = client

    def check_input(self, user_input: str) -> Dict:
        """Validate and sanitize user input against guardrails."""
        result = self.guard.validate("input", user_input)
        if result.valid:
            return {"passed": True, "message": user_input}
        else:
            return {
                "passed": False,
                "message": result.message,
                "reason": result.error or "Input violates content policy."
            }

    def sanitize_output(self, model_output: str) -> str:
        """Sanitize and validate model output."""
        result = self.guard.validate("output", model_output)
        if result.valid:
            return model_output
        else:
            return f"[⚠️ Output was sanitized for safety: {result.message}]"

    async def guarded_chat(self, user_input: str, chat_func) -> Dict:
        """Run a chat through input/output validation."""
        check = self.check_input(user_input)
        if not check["passed"]:
            return {
                "trace": [{"stage": "guardrails", "result": "blocked", "reason": check["reason"]}],
                "final_output": check["message"]
            }

        # Run the model
        model_response = await chat_func(user_input)

        # Sanitize the output
        clean_response = self.sanitize_output(model_response)
        return {
            "trace": [{"stage": "guardrails", "result": "passed"}],
            "final_output": clean_response
        }
