import os
import openai
from dotenv import load_dotenv

load_dotenv()

class FinR1Client:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="http://0.0.0.0:8000/v1",
            api_key=os.getenv("OPENAI_API_KEY", "dummy_key")  # Dummy key for local server
        )
        self.model = "SUFE-AIFLM-Lab/Fin-R1"

    def get_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Fin-R1, a truthful financial advisor. Provide honest investment advice based on facts, not media hype."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()