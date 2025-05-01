import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

class LLMHandler:
    def __init__(self):
        self.default_model = "groq"

    async def generate(self, prompt: str, model: str = None):
        model = model or self.default_model

        if model == "groq":
            return await self.query_groq(prompt)
        else:
            raise ValueError("Unsupported model")

    async def query_groq(self, prompt: str) -> str:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
               model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Groq SDK Error: {str(e)}")
            return f"Error talking to Groq API: {str(e)}"