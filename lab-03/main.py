import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

load_dotenv()

model_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = PineconeVectorStore(
    index=pc.Index("movies-walkthrough"),
    embedding=embeddings,
    text_key="text"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about blockbuster movies. "
     "Use only the retrieved context. If the context is insufficient, say you don't know."),
    ("human",
     "Question: {question}\n\nContext:\n{context}")
])

# --- RAG chain ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "question": lambda x: x["question"],
        "context": lambda x: format_docs(retriever.invoke(x["question"]))
    }
    | prompt
    | model_gemini
    | StrOutputParser()
)

while True:
    question = input("User: ").strip()
    if question.lower() in ["exit", "quit"]:
        break

    answer = chain.invoke({"question": question})
    print("\nAssistant:", answer)
    print()