import openai
from dotenv import load_dotenv
import os

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("GAPGPT_API_KEY"),
                       base_url="https://api.gapgpt.app/v1")

print("Connected to GAPGPT API")

response = client.chat.completions.create(
    model="gapgpt-qwen-3.5", 
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)


print(response.choices[0].message.content)

print("======================TOKEN USAGE======================")
print(f"input : {response.usage.prompt_tokens}")
print(f"output : {response.usage.completion_tokens}")