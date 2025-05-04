# # backend/app/main.py
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI, Request
# from app.mcp.mcp_server import MCPServer
# from pinecone import Pinecone
# import os
# from pydantic import BaseModel

# # Initialize FastAPI app
# app = FastAPI()

# # Setup CORS (allows frontend to access backend)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can later restrict to your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize MCP Server
# mcp_server = MCPServer()

# # Initialize Pinecone client
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# # === Models ===
# class ChatRequest(BaseModel):
#     user_input: str

# # === Routes ===

# @app.get("/")
# def root():
#     return {"message": "✅ AI Research Lab Backend is Running!"}

# @app.get("/healthcheck")
# def healthcheck():
#     """System Health Check Endpoint"""
#     health = {
#         "server": "✅ Alive",
#         "pinecone": "❓ Unknown",
#         "groq_api_key": "❓ Unknown",
#         "arxiv_accessible": "✅ Always accessible",
#     }

#     # Pinecone check
#     try:
#         index_list = pc.list_indexes()
#         health["pinecone"] = "✅ Connected" if index_list else "⚠️ No indexes found"
#     except Exception as e:
#         health["pinecone"] = f"❌ Error: {str(e)}"

#     # Groq API Key check
#     health["groq_api_key"] = "✅ Present" if os.getenv("GROQ_API_KEY") else "❌ Missing"

#     return health

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     """Main chat endpoint."""
#     user_input = request.user_input

#     if not user_input:
#         return {"response": "No user input provided."}

#     response = await mcp_server.route(user_input)
#     return {
#         "role": "assistant",
#         "content": response["final_output"],
#         "trace": response["trace"],  # ✅ Send trace to frontend for dev tools if needed
#     }

# @app.get("/stats")
# def stats():
#     """Placeholder stats endpoint (future extension)"""
#     return {
#         "chats_served": 0,   # To be implemented later
#         "papers_uploaded": 0,
#         "books_uploaded": 0
#     }