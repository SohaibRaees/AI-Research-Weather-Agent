import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print("API key loaded:", OPENROUTER_API_KEY is not None)

llm = ChatOpenAI(
    model="openrouter/free",
    temperature=0,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

response = llm.invoke(
    "What is the capital of Pakistan?"
)

print(response.content)