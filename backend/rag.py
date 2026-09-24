import os

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env"
    )

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="rag_documents"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

def add_documents(documents):

    if not documents:
        return 0

    chunks = text_splitter.split_documents(
        documents
    )

    if not chunks:
        return 0
    vector_db.add_documents(chunks)
    return len(chunks)
def get_retriever():
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)

def ask_question(question):
    question = question.strip()
    if not question:
        return {
            "answer": "Please enter a question."
        }

    retriever = get_retriever()
    documents = retriever.invoke(
        question
    )
    if not documents:
        return {
            "answer": (
                "I could not find the answer "
                "in the provided documents."
            )
        }

    context = "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content.strip()
    )

    if not context.strip():
        return {
            "answer": (
                "I could not find the answer "
                "in the provided documents."
            )
        }

    prompt = f"""
You are a RAG AI Assistant.

Answer the user's question using ONLY
the information given in the context.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. Give a clear and simple answer.
4. If the answer is not present in the context,
   say exactly:

"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(
        prompt
    )
    answer = response.content
    if isinstance(answer, list):
        parts = []
        for item in answer:
            if isinstance(item, dict):
                text = item.get(
                    "text",
                    ""
                )

                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        answer = "".join(parts)
    return {
        "answer": str(answer).strip()
    }