

# 🧠 AI Research Assistant

An intelligent research assistant powered by LLMs and LangGraph that helps you generate research proposals, explore topics, summarize papers, and more. Inspired by ChatGPT — with chat history, multi-agent reasoning, and integrated document support.

---

## 🚀 Features

- ✅ **Chat Interface** with research-focused LLM workflows
- ✅ **Multi-agent reasoning**: Planner, Researcher, Critic, Verifier, Summarizer
- ✅ **Smart Mode Switching**: Summary or Research based on query
- ✅ **LLM Integration**: Groq, HuggingFace, Gemini (extensible)
- ✅ **LangGraph**-style workflows with traceable steps
- ✅ **Chat History**: Rename, delete, and switch between sessions
- ✅ **Streamlit UI**: WhatsApp-style bubbles, avatars, PDF upload support

---

## 📁 Project Structure

```
ai_research_lab/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── agents/             # All agent logic (planner, critic, etc.)
│   │   ├── mcp/                # MCPServer (LangGraph-style orchestrator)
│   │   ├── utils/              # Embedding, LLM handler, etc.
│   │   └── schemas/            # Pydantic types
├── frontend/
│   └── app.py                  # Streamlit app interface
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai_research_lab.git
cd ai_research_lab
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file in the backend folder:

```env
GROQ_API_KEY=your-groq-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=your-index-name
```

---

## ▶️ Run the App

```bash
cd frontend
streamlit run gui.py
```

✅ This will **automatically launch the FastAPI backend** if it's not already running.

---

## 📚 Example Prompts

- `deep learning` → Gets a one-paragraph summary.
- `explain deep learning` → Detailed educational explanation.
- `deep learning advancements` → Retrieves and summarizes latest research.
- `new research on deep learning` → Proposes new research, generates plan, critique, and summary.

---

## 🧩 Tech Stack

- **LangChain + LangGraph** for reasoning flow
- **Groq LLMs** (LLaMA-3), with extensible model support
- **Pinecone** for RAG + vector search
- **FastAPI** for backend
- **Streamlit** for frontend
- **Wikipedia & arXiv APIs** for context

---

## 🧪 Future Enhancements

- 📄 PDF parsing and in-chat citations
- 🧠 Agent memory / scratchpad
- 📊 Admin dashboard for usage analytics
- 🌐 Multi-model routing (Groq vs Gemini vs HuggingFace)

---
