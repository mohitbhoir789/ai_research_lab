# frontend/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app.mcp.mcp_server import MCPServer
import uuid
import streamlit as st
import streamlit.components.v1 as components
import streamlit_javascript as stjs
import asyncio
import nest_asyncio
from dotenv import load_dotenv



load_dotenv()
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


# Configure Streamlit
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# Initialize MCP server
mcp_server = MCPServer()

# === Session State ===
if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_id = str(uuid.uuid4())
    st.session_state.active_chat = new_id
    st.session_state.chats[new_id] = {"title": "New Chat", "history": []}
if "provider" not in st.session_state:
    st.session_state.provider = "groq"
    st.session_state.model = "gpt-4"

# === Sidebar ===
with st.sidebar:
    st.markdown("## 💬 Chats")
    for chat_id, chat in st.session_state.chats.items():
        is_active = st.session_state.active_chat == chat_id
        bg_color = "#2f2f2f" if is_active else "transparent"
        st.markdown(
            f"""
            <div id="chat-{chat_id}" class="chat-item" style="padding: 8px 12px; margin-bottom: 8px; border-radius: 6px; background-color: {bg_color}; cursor: pointer;">
                {chat['title']}
            </div>
            """, unsafe_allow_html=True
        )

    components.html("""
    <script>
      const chats = window.parent.document.querySelectorAll('.chat-item');
      chats.forEach(chat => {
        chat.addEventListener('dblclick', () => {
          const id = chat.id.replace('chat-', '');
          window.parent.postMessage({type: 'rename_chat', id: id}, '*');
        });
        chat.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          const id = chat.id.replace('chat-', '');
          window.parent.postMessage({type: 'delete_chat', id: id}, '*');
        });
        chat.addEventListener('click', () => {
          const id = chat.id.replace('chat-', '');
          window.parent.postMessage({type: 'switch_chat', id: id}, '*');
        });
      });
    </script>
    """, height=0)

    event = stjs.st_javascript("window.addEventListener('message', e => { Streamlit.setComponentValue(e.data) })")
    if event:
        event_type = event.get("type")
        event_id = event.get("id")
        if event_type == "rename_chat":
            st.session_state.rename_id = event_id
        elif event_type == "delete_chat":
            del st.session_state.chats[event_id]
            if st.session_state.active_chat == event_id:
                st.session_state.active_chat = next(iter(st.session_state.chats), None)
            st.experimental_rerun()
        elif event_type == "switch_chat":
            st.session_state.active_chat = event_id
            st.experimental_rerun()

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "history": []}
        st.session_state.active_chat = new_id

    # Provider / Model Settings
    st.markdown("## ⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["groq", "openai", "gemini"],
                            index=["groq", "openai", "gemini"].index(st.session_state.provider))
    model = st.text_input("Model", value=st.session_state.model)

    if st.button("✅ Update Model"):
        mcp_server.update_model_settings(model=model, provider=provider)
        st.session_state.provider = provider
        st.session_state.model = model
        st.success(f"✅ Model set to `{model}` using `{provider}`")

    # # File Upload
    # st.markdown("## 📎 Upload Files")
    # uploaded_files = st.file_uploader("Upload PDFs, CSVs, or TXTs", type=["pdf", "csv", "txt"], accept_multiple_files=True)

# === Rename Chat UI ===
if "rename_id" in st.session_state:
    rid = st.session_state.rename_id
    if rid in st.session_state.chats:
        new_name = st.text_input("Rename Chat", value=st.session_state.chats[rid]["title"], key="rename_input")
        if new_name and new_name.strip():
            st.session_state.chats[rid]["title"] = new_name.strip()
            del st.session_state.rename_id
            st.experimental_rerun()
    else:
        del st.session_state.rename_id

# === Main Chat UI ===
st.title("🧠 AI Research Assistant")

current_chat_id = st.session_state.active_chat
current_chat = st.session_state.chats[current_chat_id]

for role, msg in current_chat["history"]:
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(msg)

st.markdown("---")

# Define asynchronous handler
async def handle_prompt(prompt):
    return await mcp_server.route(prompt)

# === Prompt Handling ===
if prompt := st.chat_input("Ask me anything..."):
    current_chat["history"].append(("user", prompt.strip()))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(handle_prompt(prompt))
                output = result.get("final_output", "❌ No response.")
                current_chat["history"].append(("assistant", output))
                st.markdown(output)
            except Exception as e:
                st.error(f"🚨 Failed to generate response: {e}")