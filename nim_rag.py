import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import re
import chromadb
from chromadb.utils import embedding_functions

# ==============================
# ⚠️ PASTE YOUR NVIDIA API KEY HERE
# ==============================
NVIDIA_API_KEY = "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"

# ==============================
# EMBEDDING + CHROMADB SETUP
# ==============================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chromadb_data")
collection = chroma_client.get_or_create_collection(name="documents")

# ==============================
# TEXT CLEANING & CHUNKING
# ==============================

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=100, overlap=20):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ==============================
# FILE LOADING
# ==============================

def load_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )

    elif ext == ".docx":
        doc = DocxDocument(filepath)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx")


def is_document_indexed(filepath):
    results = collection.get(
        where={"source": filepath},
        include=["metadatas"]
    )
    return len(results["ids"]) > 0


# ==============================
# LOAD & STORE DOCUMENT
# ==============================

def load_document(filepath):
    if is_document_indexed(filepath):
        print(f"✅ Document already indexed → Skipping: {filepath}")
        return

    print(f"📄 Loading document: {filepath}")
    text = load_text_from_file(filepath)

    chunks = chunk_text(text)

    print(f"🧠 Creating embeddings for {len(chunks)} chunks...")
    embeddings = embedder.encode(
        chunks, convert_to_numpy=True, show_progress_bar=True
    )

    ids = [f"{os.path.basename(filepath)}_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": filepath, "chunk_id": i} for i in range(len(chunks))]
    )

    print("✅ Document indexed successfully!")


# ==============================
# RETRIEVE CONTEXT FROM CHROMA
# ==============================

def retrieve_context(query, k=3):
    if collection.count() == 0:
        return "No document loaded. Please load a document first."

    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    top_chunks = results["documents"][0]
    return "\n\n".join(top_chunks)


# ==============================
# ✅ NVIDIA CLOUD NIM (CPU) CALL
# ==============================

def ask_nim(question):
    context = retrieve_context(question)

    if context.startswith("No document"):
        return context

    prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not present, respond with:
"The document does not contain that information."

Context:
{context}

Question: {question}
"""

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NVIDIA_API_KEY}"
    }

    payload = {
        "model": "meta/llama3-8b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ NVIDIA NIM Error: {e}"


# ==============================
# ✅ MAIN PROGRAM
# ==============================

if __name__ == "__main__":
    print("✅ NVIDIA NIM + RAG (CPU MODE)")
    print("-" * 50)

    filepath = input("Enter path to document (.txt / .pdf / .docx): ").strip()
    load_document(filepath)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if question:
            print("\n🤖 NIM:", ask_nim(question))

    print("✅ Goodbye!")

