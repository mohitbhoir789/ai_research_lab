# frontend/app.py

import streamlit as st
st.set_page_config(page_title="AI Research Assistant", layout="wide")

import os
import sys
import uuid
import asyncio
from dotenv import load_dotenv
import nest_asyncio
import os, sys

# Load environment variables (API keys, etc.)
load_dotenv()
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Model options for each provider
PROVIDER_MODELS = {
    "groq": [
        "llama2-70b-4096",
        "llama2-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
    ],
    "openai": [
        "gpt-4",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ],
    "gemini": [
        "gemini-pro",
        "gemini-pro-vision",
    ]
}

# Apply nest_asyncio so asyncio.run()/get_event_loop().run_until_complete works inside Streamlit
nest_asyncio.apply()

# Initialize asyncio event loop properly
def init_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

loop = init_async()

# === Session State Initialization ===
if "session_id" not in st.session_state:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app"))
    sys.path.insert(0, BASE_DIR)

    from mcp.mcp_server import MCPServer
    st.session_state.mcp_server = MCPServer()

    st.session_state.session_id = loop.run_until_complete(
        st.session_state.mcp_server.start_session()
    )

if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_id = str(uuid.uuid4())
    st.session_state.active_chat = new_id
    st.session_state.chats[new_id] = {"title": "New Chat", "history": []}

if "provider" not in st.session_state:
    st.session_state.provider = "groq"
    st.session_state.model = "llama2-70b-4096"

# === Sidebar (Chats + Settings) ===
with st.sidebar:
    st.markdown("## 💬 Chats")
    for chat_id, chat in st.session_state.chats.items():
        is_active = (st.session_state.active_chat == chat_id)
        button_label = f"{'▶️ ' if is_active else ''}{chat['title']}"
        if st.button(button_label, key=chat_id, type="secondary" if is_active else "primary"):
            st.session_state.active_chat = chat_id

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "history": []}
        st.session_state.active_chat = new_id

    st.markdown("---")
    st.markdown("## ⚙️ Model Settings")
    provider = st.selectbox(
        "LLM Provider", ["groq", "openai", "gemini"],
        index=["groq","openai","gemini"].index(st.session_state.provider)
    )
    
    # Show model dropdown based on selected provider
    model = st.selectbox(
        "Model",
        options=PROVIDER_MODELS[provider],
        index=PROVIDER_MODELS[provider].index(st.session_state.model) if st.session_state.model in PROVIDER_MODELS[provider] else 0
    )
    
    if st.button("✅ Update Model"):
        # update both MCPServer and session state
        st.session_state.mcp_server.update_model_settings(model=model, provider=provider)
        st.session_state.provider = provider
        st.session_state.model = model
        st.success(f"Model set to `{model}` with `{provider}`")

# === Main Chat Area ===
st.title("🧠 AI Research Assistant")

current = st.session_state.active_chat
history = st.session_state.chats[current]["history"]

# render past messages
for role, text in history:
    with st.chat_message(role, avatar="🧑‍💻" if role=="user" else "🤖"):
        st.markdown(text)

# handle new user prompt
if prompt := st.chat_input("Ask me anything..."):
    # record user message
    history.append(("user", prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # call MCPServer.route(...) synchronously via asyncio
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                resp = loop.run_until_complete(
                    st.session_state.mcp_server.route(
                        user_input=prompt,
                        session_id=st.session_state.session_id,
                    )
                )
                answer = resp.get("final_output", "❌ No response.")
            except Exception as e:
                answer = f"🚨 Error: {e}"

            # record and display assistant message
            history.append(("assistant", answer))
            st.markdown(answer)