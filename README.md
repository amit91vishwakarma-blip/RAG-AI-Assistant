# RAG AI Assistant 🤖

A simple AI assistant that answers questions from **PDFs, websites, and images** using Retrieval-Augmented Generation (RAG).

## Features

- PDF Question Answering
- Website Content Processing
- Image Text Extraction
- Semantic Search with ChromaDB
- Google Gemini AI
- Streamlit Chat Interface

## Tech Stack

- Python
- FastAPI
- Streamlit
- LangChain
- Google Gemini
- ChromaDB

## RAG Workflow

text
PDF / Website / Image
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embeddings
        ↓
ChromaDB
        ↓
Question
        ↓
Relevant Context
        ↓
Gemini
        ↓
Answer