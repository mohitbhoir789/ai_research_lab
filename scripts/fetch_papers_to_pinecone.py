# scripts/fetch_papers_to_pinecone.py
import sys
sys.path.append('./backend')

# scripts/fetch_papers_to_pinecone.py

import arxiv
import os
import json
from datetime import datetime
from pinecone import Pinecone
from dotenv import load_dotenv
from app.utils.embeddings import EmbeddingHandler

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

embedding_handler = EmbeddingHandler()

UPLOAD_LOG_PATH = "uploads/papers_uploaded.json"
os.makedirs(os.path.dirname(UPLOAD_LOG_PATH), exist_ok=True)

def save_upload_log(entry):
    """Save an upload record locally into JSON file."""
    try:
        if os.path.exists(UPLOAD_LOG_PATH):
            with open(UPLOAD_LOG_PATH, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)

        with open(UPLOAD_LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"⚠️ Could not update local upload log: {str(e)}")

async def fetch_and_upload_arxiv(query="computer science", max_results=50):
    """Fetch papers from Arxiv, embed them, and upload to Pinecone."""
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    uploaded_count = 0

    for result in search.results():
        paper_id = f"arxiv-{result.entry_id}"

        # Check for duplicates based on source_url
        response = index.query(
            namespace="default",
            vector=[0.0] * 384,  # dummy vector for query (assuming 384 dims)
            top_k=1,
            filter={"source_url": {"$eq": result.entry_id}},
            include_metadata=True
        )

        if response.get('matches'):
            print(f"[Skipped] Paper already uploaded: {result.title}")
            continue

        # Create embedding
        combined_text = f"{result.title}\n\n{result.summary}"
        embedding = await embedding_handler.embed_query(combined_text)

        # Create vector data
        vector_data = {
            "id": paper_id,
            "values": embedding,
            "metadata": {
                "source_name": result.title,
                "source_type": "paper",
                "source_url": result.entry_id,
                "text": result.summary
            }
        }

        # Upload
        index.upsert(
            vectors=[vector_data],
            namespace="default"
        )

        # Save locally
        save_upload_log({
            "paper_id": paper_id,
            "title": result.title,
            "uploaded_at": datetime.utcnow().isoformat()
        })

        print(f"[Uploaded] {result.title}")
        uploaded_count += 1

    print(f"\n✅ Uploaded {uploaded_count} new papers to Pinecone and logged locally.")

# Usage Example
# await fetch_and_upload_arxiv(query="computer science OR data science", max_results=30)

if __name__ == "__main__":
    import asyncio
    query = "computer science"
    max_results = 50
    asyncio.run(fetch_and_upload_arxiv(query=query, max_results=max_results))

# Note: This script fetches papers from arXiv and uploads them to Pinecone.
# Ensure you have the required libraries installed:
# pip install arxiv pinecone-client python-dotenv
# Also, ensure you have your Pinecone API key and index name set in your .env file.
# The script uses the arXiv API to fetch papers based on a query and uploads them to Pinecone.
# The embedding handler is used to convert the text into embeddings before uploading.
# The script checks for duplicates before uploading to avoid unnecessary uploads.
# The script is designed to be run as a standalone script, but can also be imported as a module.
## to run script from terminal "./run_script.sh fetch_papers_to_pinecone"