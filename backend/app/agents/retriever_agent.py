"""
Retriever Agent Module
Specialized agent for retrieving contextual knowledge from various sources.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import asyncio
from backend.app.utils.llm import LLMConfig, LLMProvider
from backend.app.agents.agent_core import BaseAgent
from backend.app.utils.embeddings import EmbeddingHandler
from backend.app.utils.llm import LLMHandler
from backend.app.utils.guardrails import GuardrailsChecker
import logging
from pinecone import Pinecone
import wikipedia
import arxiv
from dotenv import load_dotenv

import requests
from io import BytesIO
from PyPDF2 import PdfReader

load_dotenv()
logger = logging.getLogger(__name__)

class RetrieverAgent(BaseAgent):
    """Agent specialized in retrieving relevant contextual knowledge from static and live sources."""

    def __init__(self,
                 model: str = "gemini-2.0-flash",
                 provider: str = "gemini",
                 agent_id: str = None,
                 **kwargs):
        
        # Initialize handlers
        from backend.app.utils.llm import LLMHandler
        from backend.app.utils.guardrails import GuardrailsChecker
        
        self.llm = LLMHandler()
        self.guardrails = GuardrailsChecker()
        self.temperature = 0.4
        self.max_tokens = 1000

        # Initialize base agent
        super().__init__(
            model=model,
            provider=provider,
            agent_id=agent_id
        )

        self.llm_config = LLMConfig(
            model=self.model,
            provider=LLMProvider(self.provider),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # External tools and services
        self.embedding_handler = EmbeddingHandler()
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

        # Extended system prompt
        specialized_prompt = """
        As a Retriever Agent and expert librarian, your role is to curate and provide accurate, well-organized background context on any Computer Science or Data Science topic.

        You will act like a knowledgeable librarian, drawing from:
        - Retrieved static documents in the vector database (books, papers, etc.)
        - Concise Wikipedia summaries
        - Up-to-date academic papers via arXiv

        Always:
        - Label sections clearly (e.g., 'Book Context', 'Paper Context', 'Wikipedia Summary').
        - Refuse non-Computer Science/Data Science topics (reinforce guardrails).
        - Avoid making assumptions when data is unavailable; indicate "No X context found."
        """
        # Add specialized prompt to system messages
        self.add_message("system", specialized_prompt)

    async def run(self, query: str) -> str:
        """
        Unified entry point for the retriever agent. 
        Fetches book, paper, and Wikipedia context in order:
        1. Book from Pinecone
        2. Paper: first from Pinecone, if not found then from arXiv (and upload)
        3. Wikipedia summary
        """

        # Step 1: Retrieve book context from Pinecone
        book_context = await self.retrieve_static_context(query, source_type="book")

        # Step 2: Retrieve paper context from Pinecone
        paper_context = await self.retrieve_static_context(query, source_type="paper")

        # If no paper context, fetch from arXiv and upload to Pinecone
        if not paper_context:
            paper_context, vectors = await self.fetch_and_store_arxiv(query)
            # Schedule background upsert
            asyncio.create_task(self._upsert_vectors(vectors))
        else:
            vectors = []

        # Step 3: Always fetch Wikipedia summary
        wikipedia_context = self.fetch_wikipedia_summary(query)

        combined_context = f"""## 📘 Book Context

{book_context or "No book context found."}

---

## 🧪 Paper Context

{paper_context or "No paper context found."}

---

## 🌐 Wikipedia Summary

{wikipedia_context or "No Wikipedia summary available."}
"""

        # Optional: Use LLM to summarize or post-process the combined context
        # response = await self.llm.generate(combined_context, config=self.llm_config)
        # return response.text

        return combined_context

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

    async def fetch_and_store_arxiv(self, query: str, max_results: int = 3) -> tuple[str, list]:
        """
        Fetch full PDFs from arXiv, extract text, embed in chunks, and store into Pinecone.
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
                url = result.entry_id
                pdf_url = result.pdf_url

                # Download PDF
                resp = requests.get(pdf_url)
                resp.raise_for_status()
                reader = PdfReader(BytesIO(resp.content))

                # Extract full text
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() or ""

                # Chunk text into ~1000 character segments
                chunk_size = 1000
                for i in 0, len(full_text), chunk_size:
                    chunk = full_text[i:i+chunk_size]
                    embedding = await self.embedding_handler.embed_query(chunk)
                    vectors.append({
                        "id": f"arxiv-{result.entry_id}-chunk-{i//chunk_size}",
                        "values": embedding,
                        "metadata": {
                            "source_type": "paper",
                            "source_name": title,
                            "source_url": url,
                            "chunk_index": i//chunk_size,
                            "text": chunk[:200]  # store first 200 chars as metadata snippet
                        }
                    })

                # For user-facing summary, use abstract
                summary = result.summary
                summaries.append(f"- **{title}**\n{summary}\n🔗 {url}\n")

            return "\n".join(summaries), vectors

        except Exception as e:
            logger.error(f"ArXiv PDF fetch/store failed: {e}")
            return "⚠️ Error fetching and storing PDF data from arXiv.", []

    # async def retrieve_full_context(self, query: str) -> RetrieverOutput:
    #     """
    #     Return all sources as structured schema output.

    #     Args:
    #         query: Search query

    #     Returns:
    #         RetrieverOutput object
    #     """
    #     book = await self.retrieve_static_context(query, source_type="book")
    #     paper = await self.retrieve_static_context(query, source_type="paper")
    #     if not paper or len(paper) < 500:
    #         paper += await self.fetch_and_store_arxiv(query)
    #     wiki = self.fetch_wikipedia_summary(query)

    #     return RetrieverOutput(
    #         book_context=book,
    #         paper_context=paper,
    #         wikipedia_context=wiki
    #     )

    async def _upsert_vectors(self, vectors):
        """Background task to upsert embeddings into Pinecone."""
        try:
            if vectors:
                self.index.upsert(vectors=vectors, namespace="default")
            logger.info("Background upsert to Pinecone completed.")
        except Exception as e:
            logger.error(f"Background Pinecone upsert failed: {e}")


# For testing the agent directly
if __name__ == "__main__":
    import asyncio

    async def test_agent():
        agent = RetrieverAgent()
        query = "what is logistic regression"
        print("Running retriever agent...")
        result = await agent.run(query)
        print(result)

    asyncio.run(test_agent())