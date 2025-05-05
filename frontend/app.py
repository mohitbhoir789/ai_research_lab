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

# === CSS for Chat GPT like styling ===
st.markdown("""
<style>
    /* ChatGPT-like styling */
    .main {
        background-color: #ffffff;
    }
    .stSidebar {
        background-color: #202123;
        color: white;
    }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        padding: 10px 15px;
        margin: 2px 0px;
        border-radius: 5px;
        border: none;
        background-color: transparent;
        color: white;
        display: flex;
        align-items: center;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: #2A2B32;
    }
    /* Enhanced active chat styling */
    section[data-testid="stSidebar"] .active-chat button {
        background-color: #343541 !important;
        border-left: 3px solid #10a37f !important;
        font-weight: bold;
    }
    /* Chat options styling */
    .chat-options {
        display: inline-block;
        margin-left: 5px;
        cursor: pointer;
    }
    /* Chat title update styling */
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
        gap: 0;
    }
    /* Chat container styling */
    .chat-container {
        height: 600px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    /* Delete button styling */
    .delete-btn {
        color: #ff4b4b;
        background-color: transparent;
        border: none;
        cursor: pointer;
    }
    /* Chat row styling */
    .chat-row {
        display: flex;
        align-items: center;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .chat-row:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .chat-title {
        flex-grow: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .chat-controls {
        display: flex;
        align-items: center;
    }
    /* Remove extra space at the top of sidebar */
    section[data-testid="stSidebar"] > div {
        padding-top: 0rem;
    }
    section[data-testid="stSidebar"] h2:first-of-type {
        margin-top: 0;
        padding-top: 0.5rem;
    }
    /* Collapse button styling */
    .collapse-button {
        cursor: pointer;
        margin-left: 6px;
        padding: 0 4px;
        font-size: 14px;
        border-radius: 4px;
        vertical-align: middle;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .collapse-button:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    /* Hide Chats toggle button styling */
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    /* Chat list container when collapsed */
    .chat-list-collapsed {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease;
    }
    /* Chat list container when expanded */
    .chat-list-expanded {
        max-height: 1000px;
        transition: max-height 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

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
    st.session_state.chats[new_id] = {"title": "New Chat", "history": [], "mode": "chat"}

if "provider" not in st.session_state:
    st.session_state.provider = "groq"
    st.session_state.model = "llama2-70b-4096"

if "mode" not in st.session_state:
    st.session_state.mode = "chat"

if "edit_chat_id" not in st.session_state:
    st.session_state.edit_chat_id = None
    
if "double_clicked" not in st.session_state:
    st.session_state.double_clicked = {}

if "collapsed_chats" not in st.session_state:
    st.session_state.collapsed_chats = False

# Functions for chat management
def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        # If we deleted the active chat, set a new active chat
        if st.session_state.active_chat == chat_id:
            if st.session_state.chats:
                st.session_state.active_chat = list(st.session_state.chats.keys())[0]
            else:
                # If no chats left, create a new one
                new_id = str(uuid.uuid4())
                st.session_state.active_chat = new_id
                st.session_state.chats[new_id] = {"title": "New Chat", "history": [], "mode": st.session_state.mode}

def rename_chat(chat_id):
    # Only set edit_chat_id if the chat still exists
    if chat_id in st.session_state.chats:
        st.session_state.edit_chat_id = chat_id
    else:
        # Clear edit_chat_id if the chat doesn't exist
        st.session_state.edit_chat_id = None

def handle_title_click(chat_id):
    """Handle single or double click on chat title"""
    current_time = asyncio.get_event_loop().time()
    
    # Initialize if this is the first click for this chat
    if chat_id not in st.session_state.double_clicked:
        st.session_state.double_clicked[chat_id] = {"last_click": current_time, "count": 1}
    else:
        # Check if this is a double click (within 0.5 seconds)
        time_diff = current_time - st.session_state.double_clicked[chat_id]["last_click"]
        if time_diff < 0.5:  # Double click threshold
            st.session_state.double_clicked[chat_id]["count"] += 1
        else:
            st.session_state.double_clicked[chat_id]["count"] = 1
        
        st.session_state.double_clicked[chat_id]["last_click"] = current_time
    
    # If double clicked, enter edit mode
    if st.session_state.double_clicked[chat_id]["count"] >= 2:
        rename_chat(chat_id)
        # Reset counter
        st.session_state.double_clicked[chat_id]["count"] = 0
    
    # Set as active chat on any click
    st.session_state.active_chat = chat_id

# === Sidebar (Chats + Settings) ===
with st.sidebar:
    # My Chats section with collapse toggle
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown("## 💬 My Chats")
    with col2:
        # Toggle button for collapsing chat list
        collapse_icon = "▼" if not st.session_state.collapsed_chats else "▲"
        st.markdown(f"""
        <div class="collapse-button" onclick="
            (function(){{
                const collapsed = {str(st.session_state.collapsed_chats).lower()};
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: !collapsed
                }}, '*');
            }})()
        ">{collapse_icon}</div>
        """, unsafe_allow_html=True)

    st.markdown("<small>Double-click on any chat name to edit it</small>", unsafe_allow_html=True)
    
    # Listen for collapse toggle event
    collapse_state = st.empty()
    if collapse_state.button(
        "Hide Chats" if not st.session_state.collapsed_chats else "Show Chats", 
        key="toggle_chats", 
        help="Collapse/Expand chats", 
        on_click=lambda: setattr(st.session_state, "collapsed_chats", not st.session_state.collapsed_chats)
    ):
        pass  # The on_click handles the state change
    
    # Chat list container with conditional styling
    chat_list_class = "chat-list-collapsed" if st.session_state.collapsed_chats else "chat-list-expanded"
    st.markdown(f"<div class='{chat_list_class}'>", unsafe_allow_html=True)
    
    # Only show chats when not collapsed
    if not st.session_state.collapsed_chats:
        # Render each chat with options
        for chat_id, chat in st.session_state.chats.items():
            is_active = (st.session_state.active_chat == chat_id)
            mode_icon = "💬" if chat.get("mode", "chat") == "chat" else "🔬"
            
            # Apply active chat styling
            if is_active:
                st.markdown("<div class='active-chat'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([0.85, 0.15])
            
            # Chat selection button - make it look like a clickable title
            with col1:
                chat_title = chat['title']
                if len(chat_title) > 20:
                    display_title = f"{chat_title[:20]}..."
                else:
                    display_title = chat_title
                    
                button_label = f"{mode_icon} {display_title}"
                if st.button(
                    button_label,
                    key=f"chat_{chat_id}",
                    help="Click to select chat, double-click to rename",
                    on_click=handle_title_click,
                    args=(chat_id,)
                ):
                    st.session_state.mode = chat.get("mode", "chat")
            
            # Delete button
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}", help="Delete this chat"):
                    delete_chat(chat_id)
                    st.rerun()
            
            if is_active:
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # New Chat Button below the chat list
    if st.button("+ New Chat", type="primary", key="new_chat_button"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title": "New Chat",
            "history": [],
            "mode": st.session_state.mode
        }
        st.session_state.active_chat = new_id
        st.rerun()
    
    # Chat rename form
    if st.session_state.edit_chat_id and st.session_state.edit_chat_id in st.session_state.chats:
        with st.form(key=f"rename_form_{st.session_state.edit_chat_id}"):
            chat_id = st.session_state.edit_chat_id
            current_title = st.session_state.chats[chat_id]["title"]
            new_title = st.text_input("New title", value=current_title, key=f"new_title_{chat_id}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save"):
                    st.session_state.chats[chat_id]["title"] = new_title
                    st.session_state.edit_chat_id = None
                    st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.edit_chat_id = None
                    st.rerun()

    st.markdown("---")
    st.markdown("## 🤖 Mode Selection")
    
    # Mode selection with radio buttons
    mode = st.radio(
        "Choose Mode",
        ["💬 Chat Mode", "🔬 Research Mode"],
        index=0 if st.session_state.mode == "chat" else 1,
        help="""
        Chat Mode: Get quick summaries or detailed explanations about CS/DS topics
        Research Mode: Explore research papers, propose new research ideas
        """
    )
    
    # Update mode in session state and current chat
    new_mode = "chat" if "Chat" in mode else "research"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.chats[st.session_state.active_chat]["mode"] = new_mode
    
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
current_mode = "💬 Chat Mode" if st.session_state.mode == "chat" else "🔬 Research Mode"
st.title(f"🧠 AI Research Assistant")

# Show current chat title and model info in a header
current = st.session_state.active_chat
current_chat = st.session_state.chats[current]
st.markdown(f"**Current chat:** {current_chat['title']} ({current_mode}) | Model: {st.session_state.model} ({st.session_state.provider})")

if st.session_state.mode == "chat":
    st.markdown("""
    **Chat Mode**: Ask questions about Computer Science or Data Science topics.
    - For quick answers, just ask your question
    - For detailed explanations, include words like "explain" or "in detail"
    """)
else:
    st.markdown("""
    **Research Mode**: Explore or propose academic research.
    - Search existing papers and get detailed summaries
    - Generate new research proposals
    - Get interdisciplinary perspectives
    """)

history = st.session_state.chats[current]["history"]

# Create a container for chat messages with scrolling
chat_container = st.container()
with chat_container:
    # render past messages
    for role, text in history:
        with st.chat_message(role, avatar="🧑‍💻" if role=="user" else "🤖"):
            st.markdown(text)

# handle new user prompt
placeholder_text = (
    "Ask about CS/DS topics..." if st.session_state.mode == "chat" 
    else "Explore research topics, papers, or propose new research..."
)

if prompt := st.chat_input(placeholder_text):
    # If this is the first message and the title is still "New Chat", update the title
    if len(history) == 0 and current_chat["title"] == "New Chat":
        # Generate a more descriptive title from the first message
        # Remove question words and common prefixes for cleaner titles
        clean_prompt = prompt.strip()
        
        # List of prefixes to remove for cleaner titles
        prefixes_to_remove = [
            "can you ", "could you ", "please ", "tell me ", "explain ", 
            "what is ", "how to ", "why is ", "when did ", "where is ",
            "what are ", "how do ", "what was ", "how does "
        ]
        
        for prefix in prefixes_to_remove:
            if clean_prompt.lower().startswith(prefix):
                clean_prompt = clean_prompt[len(prefix):]
                break
        
        # Capitalize first letter and trim to appropriate length
        if clean_prompt:
            clean_prompt = clean_prompt[0].upper() + clean_prompt[1:]
            
        # Add ellipsis if title is too long
        if len(clean_prompt) > 40:
            new_title = clean_prompt[:40] + "..."
        else:
            new_title = clean_prompt
            
        # Set the new title
        st.session_state.chats[current]["title"] = new_title
    
    # record user message
    history.append(("user", prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # call MCPServer.route(...) synchronously via asyncio
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..." if st.session_state.mode == "chat" else "Researching..."):
            try:
                resp = loop.run_until_complete(
                    st.session_state.mcp_server.route(
                        user_input=prompt,
                        session_id=st.session_state.session_id,
                        mode=st.session_state.mode
                    )
                )
                answer = resp.get("final_output", "❌ No response.")
            except Exception as e:
                answer = f"🚨 Error: {e}"

            # record and display assistant message
            history.append(("assistant", answer))
            st.markdown(answer)