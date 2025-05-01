import streamlit as st
import uuid
st.set_page_config(page_title="AI Research Assistant", layout="wide")
import requests
import subprocess
import time
import os
import streamlit.components.v1 as components
import streamlit_javascript as stjs

# --- Start FastAPI backend server ---
def start_backend():
    try:
        res = requests.get("http://localhost:8000/healthcheck", timeout=2)
        if res.status_code == 200:
            return  # Already running
    except requests.exceptions.RequestException as e:
        st.info("🟡 Backend not detected, attempting to start...")

    backend_path = os.path.join(os.path.dirname(__file__), "../backend")
    try:
        backend_process = subprocess.Popen(
            ["uvicorn", "app.main:app", "--reload"],
            cwd=backend_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        return backend_process
    except Exception as e:
        st.error(f"❌ Failed to start backend server: {e}")
        raise

# Call once at startup
backend_process = start_backend()


if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_id = str(uuid.uuid4())
    st.session_state.active_chat = new_id
    st.session_state.chats[new_id] = {"title": "New Chat", "history": []}


# Sidebar chat management
with st.sidebar:
    st.markdown("## 💬 Chats")
    chat_ids = list(st.session_state.chats.keys())
    for chat_id in chat_ids:
        chat = st.session_state.chats[chat_id]
        is_active = st.session_state.active_chat == chat_id
        bg_color = "#2f2f2f" if is_active else "transparent"
        st.markdown(
            f"""
            <div id="chat-{chat_id}" class="chat-item" style="padding: 8px 12px; margin-bottom: 8px; border-radius: 6px; background-color: {bg_color}; cursor: pointer;">
                {chat['title']}
            </div>
            """, unsafe_allow_html=True)

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
        if event.get("type") == "rename_chat":
            st.session_state.rename_id = event.get("id")
        elif event.get("type") == "delete_chat":
            del st.session_state.chats[event.get("id")]
            if st.session_state.active_chat == event.get("id"):
                st.session_state.active_chat = next(iter(st.session_state.chats), None)
            st.experimental_rerun()
        elif event.get("type") == "switch_chat":
            st.session_state.active_chat = event.get("id")
            st.experimental_rerun()

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "history": []}
        st.session_state.active_chat = new_id


if "rename_id" in st.session_state:
    rid = st.session_state.rename_id
    if rid in st.session_state.chats:
        new_name = st.text_input("Rename Chat", value=st.session_state.chats[rid]["title"], key="rename_input")
        if new_name and new_name.strip():
            st.session_state.chats[rid]["title"] = new_name.strip()
            del st.session_state.rename_id
            st.experimental_rerun()
        elif new_name == "":
            # Prevent empty name, keep rename input open
            pass
    else:
        del st.session_state.rename_id


st.title("🧠 AI Research Assistant")

current_chat_id = st.session_state.active_chat
current_chat = st.session_state.chats.get(current_chat_id, {"history": []})
for role, msg in current_chat["history"]:
    if role == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg)
st.markdown("---")

if prompt := st.chat_input("Ask me anything..."):
    current_chat["history"].append(("user", prompt.strip()))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"user_input": prompt}
                )
                if response.ok:
                    result = response.json()
                    output = result.get("content") or result.get("final_output", "❌ No response.")
                    if output:
                        current_chat["history"].append(("assistant", output))
                        st.markdown(output)
                    else:
                        st.markdown("⚠️ No content returned.")
                else:
                    st.error(f"❌ Server error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚨 Failed to connect to backend: {e}")