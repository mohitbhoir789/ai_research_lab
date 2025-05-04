"""
Retriever Agent Module
Specialized agent for retrieving contextual knowledge from various sources.
"""

import logging
from backend.app.agents.agent_core import LLMAgent
from backend.app.utils.embeddings import EmbeddingHandler
from backend.app.schemas.agent_schemas import RetrieverOutput
from pinecone import Pinecone
import wikipedia
import arxiv
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class RetrieverAgent(LLMAgent):
    """Agent specialized in retrieving relevant contextual knowledge from static and live sources."""

    def __init__(self,
                 name: str = "Retriever Agent",
                 model: str = "llama3-70b-8192",
                 temperature: float = 0.4,
                 max_tokens: int = 1000,
                 provider: str = "groq",
                 **kwargs):

        description = "I specialize in retrieving relevant background context from books, research papers, and the web."

        super().__init__(
            name=name,
            description=description,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            **kwargs
        )

        # External tools and services
        self.embedding_handler = EmbeddingHandler()
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

        # Extended system prompt
        specialized_prompt = """
        As a Retriever Agent, your job is to gather and return accurate and relevant background knowledge.

        Your information sources may include:
        - Retrieved static documents from vector DB (books, papers, etc.)
        - Wikipedia summaries
        - Recent academic papers via arXiv

        Always label each section clearly (e.g., 'Book Context', 'Wikipedia Summary') and avoid assumptions if no data is found.
        """
        self.update_system_prompt(self.system_prompt + specialized_prompt)

    async def run(self, query: str) -> str:
        """
        Unified entry point for the retriever agent. Fetches book, arXiv, and Wikipedia context.

        Args:
            query: The research topic or question

        Returns:
            Combined context string
        """
        book_context = await self.retrieve_static_context(query, source_type="book")

        if not book_context or len(book_context) < 500:
            paper_context = await self.fetch_and_store_arxiv(query)
        else:
            paper_context = await self.retrieve_static_context(query, source_type="paper")

        wikipedia_context = self.fetch_wikipedia_summary(query)

        return f"""## 📘 Book Context

{book_context or "No book context found."}

---

## 🧪 Paper Context

{paper_context or "No paper context found."}

---

## 🌐 Wikipedia Summary

{wikipedia_context or "No Wikipedia summary available."}
"""

    async def retrieve_static_context(self, query: str, source_type: str = "book", top_k: int = 5) -> str:
        """
        Retrieve static embedded content from Pinecone vector DB.

        Args:
            query: Search query
            source_type: "book" or "paper"
            top_k: Number of top matches

        Returns:
            Retrieved formatted context
        """
        query_vector = await self.embedding_handler.embed_query(query)

        try:
            response = self.index.query(
                namespace="default",
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter={"source_type": {"$eq": source_type}}
            )
        except Exception as e:
            logger.error(f"Vector DB retrieval failed: {e}")
            return ""

        results = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            source = metadata.get("source_name", "Unknown Source")
            text = metadata.get("text", "")
            results.append(f"- **{source}**\n{text}")

        return "\n\n".join(results)

    def fetch_wikipedia_summary(self, query: str, sentences: int = 5) -> str:
        """
        Fetch summary from Wikipedia.

        Args:
            query: Topic to search
            sentences: Number of sentences to return

        Returns:
            Summary string
        """
        try:
            summary = wikipedia.summary(query, sentences=sentences)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            return f"⚠️ Disambiguation error. Options: {', '.join(e.options[:5])}..."
        except wikipedia.exceptions.PageError:
            return "⚠️ Page not found on Wikipedia."
        except Exception as e:
            logger.error(f"Wikipedia error: {e}")
            return "⚠️ Error fetching Wikipedia summary."

    async def fetch_and_store_arxiv(self, query: str, max_results: int = 3) -> str:
        """
        Fetch papers from arXiv, embed, and store into Pinecone.

        Args:
            query: Search query
            max_results: Max papers to retrieve

        Returns:
            Combined summary of arXiv results
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        summaries = []
        vectors = []

        try:
            for result in search.results():
                title = result.title
                summary = result.summary
                url = result.entry_id
                full_text = f"{title}\n\n{summary}"

                embedding = await self.embedding_handler.embed_query(full_text)

                vectors.append({
                    "id": f"arxiv-{result.entry_id}",
                    "values": embedding,
                    "metadata": {
                        "source_type": "paper",
                        "source_name": title,
                        "source_url": url,
                        "text": summary
                    }
                })

                summaries.append(f"- **{title}**\n{summary}\n🔗 {url}\n")

            if vectors:
                self.index.upsert(vectors=vectors, namespace="default")

            return "\n".join(summaries)

        except Exception as e:
            logger.error(f"ArXiv fetch/store failed: {e}")
            return "⚠️ Error fetching data from arXiv."

    async def retrieve_full_context(self, query: str) -> RetrieverOutput:
        """
        Return all sources as structured schema output.

        Args:
            query: Search query

        Returns:
            RetrieverOutput object
        """
        book = await self.retrieve_static_context(query, source_type="book")
        paper = await self.retrieve_static_context(query, source_type="paper")
        if not paper or len(paper) < 500:
            paper += await self.fetch_and_store_arxiv(query)
        wiki = self.fetch_wikipedia_summary(query)

        return RetrieverOutput(
            book_context=book,
            paper_context=paper,
            wikipedia_context=wiki
        )


# For testing the agent directly
if __name__ == "__main__":
    import asyncio

    async def test_agent():
        agent = RetrieverAgent()
        query = "Transformer neural networks in NLP"
        result = await agent.run(query)
        print(result)

    asyncio.run(test_agent())