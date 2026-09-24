import os
import shutil
import base64

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import add_documents, ask_question
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

app = FastAPI(
    title="RAG AI Assistant",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vision_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)

@app.get("/")
def home():

    return {
        "message": "RAG AI Assistant Backend is running"
    }

@app.post("/ingest/pdf")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for document in documents:
            document.metadata["source_type"] = "pdf"
            document.metadata["source"] = file.filename

        chunks = add_documents(documents)

        return {
            "success": True,
            "message": "PDF processed successfully",
            "filename": file.filename,
            "pages": len(documents),
            "chunks_added": chunks
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

class URLRequest(BaseModel):
    url: str

@app.post("/ingest/url")
def ingest_url(data: URLRequest):
    url = data.url.strip()
    if not url.startswith(("http://", "https://")):

        raise HTTPException(
            status_code=400,
            detail="Please enter a valid website URL."
        )

    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        if not documents:

            raise HTTPException(
                status_code=400,
                detail="No content found on website."
            )

        for document in documents:

            document.metadata["source_type"] = "website"
            document.metadata["source"] = url

        chunks = add_documents(documents)

        return {
            "success": True,
            "message": "Website processed successfully",
            "url": url,
            "chunks_added": chunks
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Website processing failed: {str(e)}"
        )
    
@app.post("/ingest/image")
async def upload_image(file: UploadFile = File(...)):

    allowed = {
        "image/png",
        "image/jpeg",
        "image/webp"
    }

    if file.content_type not in allowed:

        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG, JPEG and WEBP allowed."
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Image is empty."
            )

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        prompt = """
Analyze this image.

Extract important visible information,
including text, headings, names, numbers,
tables and labels.

Return only the information visible
in the image. Do not invent information.
"""

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{file.content_type};"
                            f"base64,{image_base64}"
                        )
                    }
                }
            ]
        )

        response = vision_llm.invoke([message])
        image_text = response.content
        if isinstance(image_text, list):
            parts = []
            for item in image_text:
                if isinstance(item, dict):
                    text = item.get("text", "")

                    if text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            image_text = "\n".join(parts)
        image_text = str(image_text).strip()
        if not image_text:

            raise HTTPException(
                status_code=400,
                detail="Could not read image."
            )

        document = Document(
            page_content=image_text,
            metadata={
                "source_type": "image",
                "source": file.filename
            }
        )

        chunks = add_documents([document])
        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        return {
            "success": True,
            "message": "Image processed successfully",
            "filename": file.filename,
            "chunks_added": chunks
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )

class QuestionRequest(BaseModel):
    question: str
@app.post("/chat")
def chat(data: QuestionRequest):
    question = data.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = ask_question(question)
        return {
            "answer": result["answer"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {str(e)}"
        )