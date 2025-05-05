# backend/app/utils/embeddings.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import path

load_dotenv()

class EmbeddingHandler:
    def __init__(self):
        # Initialize the LangChain HuggingFace Embeddings
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name
        )

    async def embed_query(self, text: str):
        """Embed a single query string into a vector."""
        try:
            vector = self.embedder.embed_query(text)
            return vector
        except Exception as e:
            raise Exception(f"Embedding Error: {str(e)}")