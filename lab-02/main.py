from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

country = input("Enter a country name: ")
template = PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}?"
)

model_qwen = ChatOpenAI(
    model="gapgpt-qwen-3.5",
    openai_api_key=os.getenv("GAPGPT_API_KEY"),
    openai_api_base="https://api.gapgpt.app/v1"
)
model_gemini = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

print(">> Qwen Model Response:")
response = model_qwen.invoke(template.format(country=country))
print(response.content)

print(">> Gemini Model Response:")
response = model_gemini.invoke(template.format(country=country))
print(response)
