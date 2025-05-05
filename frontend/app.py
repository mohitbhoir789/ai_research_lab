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

# Apply nest_asyncio so asyncio.run()/get_event_loop().run_until_complete works inside Streamlit
nest_asyncio.apply()


# === Session State Initialization ===
if "session_id" not in st.session_state:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app"))
    sys.path.insert(0, BASE_DIR)

    from mcp.mcp_server import MCPServer
    st.session_state.mcp_server = MCPServer()

    st.session_state.session_id = asyncio.get_event_loop().run_until_complete(
        st.session_state.mcp_server.start_session()
    )

if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_id = str(uuid.uuid4())
    st.session_state.active_chat = new_id
    st.session_state.chats[new_id] = {"title": "New Chat", "history": []}

if "provider" not in st.session_state:
    st.session_state.provider = "groq"
    st.session_state.model = "gpt-4"

# === Sidebar (Chats + Settings) ===
with st.sidebar:
    st.markdown("## 💬 Chats")
    for chat_id, chat in st.session_state.chats.items():
        is_active = (st.session_state.active_chat == chat_id)
        bg = "#2f2f2f" if is_active else "transparent"
        if st.button(chat["title"], key=chat_id, 
                     style={"background-color": bg, "width": "100%"}):
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
    model = st.text_input("Model", value=st.session_state.model)
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
                resp = asyncio.get_event_loop().run_until_complete(
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