#!/usr/bin/env python3

#!/usr/bin/env python3

import os
import argparse
import asyncio
import sys
import logging
from datetime import datetime

# Force PYTHONPATH to ./backend
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Now imports will work properly
from scripts.fetch_papers_to_pinecone import fetch_and_upload_arxiv
from scripts.upload_books_to_pinecone import upload_book_pdf

# Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
logfile = os.path.join(LOG_DIR, f"runner_{today}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(
        description="🚀 AI Research Lab - Ingestion Runner"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["upload_papers", "upload_book"],
        help="Choose the task to run: upload_papers or upload_book"
    )

    parser.add_argument("--query", type=str, help="Query for research papers (upload_papers)")
    parser.add_argument("--max_results", type=int, default=30, help="Max results for papers (upload_papers)")
    parser.add_argument("--file_path", type=str, help="Path to book PDF (upload_book)")
    parser.add_argument("--book_title", type=str, help="Title of the book (upload_book)")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for book upload (upload_book)")

    args = parser.parse_args()

    try:
        if args.task == "upload_papers":
            if not args.query:
                args.query = "computer science OR data science"
            logging.info(f"📄 Starting upload_papers | Query: {args.query} | Max Results: {args.max_results}")
            asyncio.run(fetch_and_upload_arxiv(query=args.query, max_results=args.max_results))
            logging.info(f"✅ Completed upload_papers")

        elif args.task == "upload_book":
            if not args.file_path or not args.book_title:
                logging.error("❌ For upload_book, --file_path and --book_title are required.")
                sys.exit(1)
            logging.info(f"📚 Starting upload_book | File: {args.file_path} | Title: {args.book_title} | Chunk Size: {args.chunk_size}")
            asyncio.run(upload_book_pdf(file_path=args.file_path, book_title=args.book_title, chunk_size=args.chunk_size))
            logging.info(f"✅ Completed upload_book")

    except Exception as e:
        logging.exception(f"🔥 Exception occurred during task: {str(e)}")

if __name__ == "__main__":
    main()


# Note: This script is designed to be run from the command line.
# It allows you to upload research papers or books to Pinecone.
# Ensure you have the required libraries installed
# pip install arxiv pinecone-client python-dotenv PyMuPDF
# Also, ensure you have your Pinecone API key and index name set in your .env file.
# Usage:
# python runner.py --task upload_papers --query "machine learning" --max_results 50
# python runner.py --task upload_book --file_path "/path/to/book.pdf" --book_title "My Book Title"
# This script fetches papers from arXiv and uploads them to Pinecone.
# It also uploads a book PDF to Pinecone in chunks.