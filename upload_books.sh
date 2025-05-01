#!/bin/bash

# upload_books.sh
# 🚀 Simple executable to upload a Book PDF and track locally

echo "🚀 Starting Book Upload and Tracking..."

# Activate virtual environment (optional if you use venv)
# source ./venv/bin/activate

# Read file path and book title from user
read -p "Enter the path to your Book PDF (e.g., ./my_books/deep_learning.pdf): " BOOK_PATH
read -p "Enter the Book Title: " BOOK_TITLE
read -p "Optional: Enter Chunk Size (default 500): " CHUNK_SIZE

# If no chunk size entered, default to 500
if [ -z "$CHUNK_SIZE" ]; then
  CHUNK_SIZE=500
fi

# Run the runner with the provided inputs
python runner.py --task upload_book --file_path "$BOOK_PATH" --book_title "$BOOK_TITLE" --chunk_size "$CHUNK_SIZE"

# Inform user
echo "✅ Upload complete!"
echo "📄 (Optional) Book upload tracking can be added into uploads/books_uploaded.json if needed."


# Note: Ensure that the runner.py script is set up to handle the upload_book task and the parameters passed.
#/upload_books.sh
#chmod +x upload_books.sh
#./upload_books.sh