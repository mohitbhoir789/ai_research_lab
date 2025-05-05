"""
Embeddings Module
Handles text embedding generation using Sentence Transformers and vector storage with Pinecone.
"""

import os
import asyncio
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()
logger = logging.getLogger(__name__)

class EmbeddingHandler:
    """Handles text embedding generation and vector storage."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        pinecone_index_name: str = None
    ):
        """Initialize embedding model and vector store."""
        # Initialize sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = pinecone_index_name or os.getenv("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            # Run embedding generation in a thread pool to avoid blocking
            embedding = await asyncio.to_thread(
                self.model.encode,
                text,
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to embed in parallel
            
        Returns:
            List of embedding vectors
        """
        try:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Run batch embedding in thread pool
                batch_embeddings = await asyncio.to_thread(
                    self.model.encode,
                    batch,
                    batch_size=batch_size,
                    convert_to_numpy=True
                )
                all_embeddings.extend(batch_embeddings.tolist())
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "default"
    ) -> bool:
        """
        Upsert vectors to Pinecone index.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and optional 'metadata'
            namespace: Pinecone namespace
            
        Returns:
            Boolean indicating success
        """
        try:
            self.index.upsert(vectors=vectors, namespace=namespace)
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False

    async def query_vectors(
        self,
        query_vector: List[float],
        namespace: str = "default",
        top_k: int = 5,
        filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Query nearest vectors from Pinecone.
        
        Args:
            query_vector: Query embedding
            namespace: Pinecone namespace
            top_k: Number of results
            filter: Optional metadata filter
            
        Returns:
            List of matches with scores and metadata
        """
        try:
            results = self.index.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_metadata=True
            )
            return results.get("matches", [])
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def find_similar(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar documents to a query using embeddings.
        
        Args:
            query: Query text
            documents: List of documents with 'text' and optional metadata
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        # Generate query embedding
        query_embedding = await self.embed_query(query)
        
        # Generate document embeddings
        doc_texts = [doc["text"] for doc in documents]
        doc_embeddings = await self.embed_batch(doc_texts)
        
        # Calculate similarities
        similarities = [
            self.cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]
        
        # Sort and return top_k results
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            doc = documents[idx].copy()
            doc["similarity"] = float(similarities[idx])
            results.append(doc)
        
        return results