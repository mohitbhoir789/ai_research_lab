#!/bin/bash

# upload_papers.sh
# 🚀 Simple executable to upload papers and track locally

echo "🚀 Starting Research Paper Upload and Tracking..."

# Activate virtual environment (optional if you use venv)
# source ./venv/bin/activate
papers="computer science" 
OR "machine learning" 
OR "artificial intelligence" 
OR "deep learning"
OR "natural language processing"
OR "reinforcement learning"
OR "computer vision"
OR "big data"
OR "cloud computing"
OR "internet of things"
OR "data mining"
OR "data visualization"
OR "data analysis"
OR "data engineering"
OR "data science ethics"
OR "data science applications"
OR "data science tools"


# Run runner.py with smart query
python runner.py --task upload_papers --query '(papers)' --max_results 100

# Inform user
echo "✅ Upload complete!"
echo "📄 Uploaded papers are logged at uploads/papers_uploaded.json"