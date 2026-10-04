import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain what retrieval-augmented generation is in one sentence."
        }
    ],
    model="openai/gpt-oss-20b"
)

print(response.choices[0].message.content)