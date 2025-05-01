# backend/app/agents/retriever_agent.py

from app.schemas.agent_schemas import RetrieverOutput
from app.utils.embeddings import EmbeddingHandler
import wikipedia
import arxiv
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

class RetrieverAgent:
    def __init__(self):
        self.embedding_handler = EmbeddingHandler()
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone.Index(self.index_name)
        
    async def run(self, input_text: str) -> str:
        """Unified run method for MCPServer chaining."""
        context = await self.retrieve_static_context(input_text, source_type="book")
        if not context or len(context) < 500:
            context += await self.fetch_and_store_arxiv(input_text)

        wikipedia_summary = self.fetch_wikipedia_summary(input_text)

        combined_context = f"📚 Book Context:\n{context}\n\n🌐 Wikipedia Summary:\n{wikipedia_summary}"
        return combined_context

    async def retrieve_static_context(self, query: str, source_type: str = "book", top_k: int = 5) -> str:
        """Retrieve matching static contexts from Pinecone."""
        query_vector = await self.embedding_handler.embed_query(query)
        response = self.index.query(
            namespace="default",
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"source_type": {"$eq": source_type}}
        )

        contexts = []
        if response and response['matches']:
            for match in response['matches']:
                metadata = match.get('metadata', {})
                text = metadata.get('text', '')
                source = metadata.get('source_name', 'Unknown Source')
                contexts.append(f"- **Source**: {source}\n{text}")

        return "\n".join(contexts)

    def fetch_wikipedia_summary(self, query: str, sentences: int = 5) -> str:
        """Fetch live Wikipedia summary."""
        try:
            summary = wikipedia.summary(query, sentences=sentences)
            return f"📚 **Wikipedia Summary:**\n\n{summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"⚠️ Disambiguation error. Options: {e.options}"
        except wikipedia.exceptions.PageError:
            return "⚠️ Wikipedia page not found."

    async def fetch_and_store_arxiv(self, query: str, max_results: int = 3) -> str:
        """Fetch research papers from arXiv, embed, and store into Pinecone."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        vectors = []

        for result in search.results():
            combined_text = f"{result.title}\n\n{result.summary}"
            embedding = await self.embedding_handler.embed_query(combined_text)

            vectors.append({
                "id": f"arxiv-{result.entry_id}",
                "values": embedding,
                "metadata": {
                    "source_type": "paper",
                    "source_name": result.title,
                    "source_url": result.entry_id,
                    "text": result.summary
                }
            })

            papers.append(f"- **Title**: {result.title}\n{result.summary}\nURL: {result.entry_id}\n")

        if vectors:
            self.index.upsert(vectors=vectors, namespace="default")

        return "\n".join(papers)

    async def retrieve_full_context(self, query: str) -> RetrieverOutput:
        """Retrieve book, paper, and Wikipedia context neatly packaged."""
        book_context = await self.retrieve_static_context(query, source_type="book")
        paper_context = await self.retrieve_static_context(query, source_type="paper")

        if not paper_context or len(paper_context) < 500:
            paper_context += await self.fetch_and_store_arxiv(query)

        wikipedia_context = self.fetch_wikipedia_summary(query)

        return RetrieverOutput(
            book_context=book_context,
            paper_context=paper_context,
            wikipedia_context=wikipedia_context
        )