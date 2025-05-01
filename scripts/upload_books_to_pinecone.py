# scripts/upload_books_to_pinecone.py
import sys
sys.path.append('./backend')
import fitz  # PyMuPDF
import os
from pinecone import Pinecone
from dotenv import load_dotenv
from app.utils.embeddings import EmbeddingHandler
import json
from datetime import datetime
from typing import List
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

embedding_handler = EmbeddingHandler()

UPLOAD_BOOKS_LOG_PATH = "uploads/books_uploaded.json"
os.makedirs(os.path.dirname(UPLOAD_BOOKS_LOG_PATH), exist_ok=True)

def save_book_upload_log(entry):
    """Save a book upload record locally into JSON file."""
    try:
        if os.path.exists(UPLOAD_BOOKS_LOG_PATH):
            with open(UPLOAD_BOOKS_LOG_PATH, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)

        with open(UPLOAD_BOOKS_LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"⚠️ Could not update local book upload log: {str(e)}")

async def upload_book_pdf(file_path: str, book_title: str, chunk_size=500):
    """Upload a book PDF into Pinecone in chunks."""
    doc = fitz.open(file_path)
    all_text = ""

    for page in doc:
        all_text += page.get_text()

    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]

    uploaded_count = 0

    for idx, chunk in enumerate(chunks):
        unique_id = f"book-{book_title.replace(' ', '_')}-{idx}"

        # Check for duplicates
        response = index.query(
            namespace="default",
            vector=[0.0] * 384,  # Assuming 384 dimensions
            top_k=1,
            filter={"source_name": {"$eq": book_title}},
            include_metadata=True
        )

        if response.get('matches'):
            print(f"[Skipped] Chunk {idx} already exists.")
            continue

        embedding = await embedding_handler.embed_query(chunk)

        index.upsert(
            vectors=[{
                "id": unique_id,
                "values": embedding,
                "metadata": {
                    "source_name": book_title,
                    "source_type": "book",
                    "text": chunk
                }
            }],
            namespace="default"
        )

        print(f"[Uploaded] Chunk {idx} of {book_title}")
        uploaded_count += 1

    # Save upload info locally
    save_book_upload_log({
        "book_title": book_title,
        "uploaded_at": datetime.utcnow().isoformat(),
        "num_chunks": uploaded_count
    })

    print(f"\n✅ Uploaded {uploaded_count} new chunks for book: {book_title} and logged locally.")

# Usage:
# await upload_book_pdf("/path/to/your/book.pdf", "Machine Learning Foundations")