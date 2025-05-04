# 🧠 AI Research Lab

An interactive Streamlit-based application for AI-driven research assistance in Computer Science and Data Science. Powered by a modular multi-agent architecture via the Model Context Protocol (MCP).

## Features

* **Streamlit Frontend**: Chat-like UI for user queries, file uploads, and session management.
* **MCPServer Orchestration**: In-process JSON-RPC server routing requests to specialized agents.
* **Agent Modules**:

  * **RetrieverAgent**: Fetches context from Pinecone, Wikipedia, or ArXiv.
  * **PlannerAgent**: Breaks queries into research plans or deep-dive steps.
  * **ResearcherAgent**: Pulls paper metadata and abstracts from ArXiv.
  * **CriticAgent**: Reviews and flags gaps in generated content.
  * **VerifierAgent**: Fact-checks claims and summaries.
  * **SummarizerAgent**: Generates concise bullet-point and paragraph summaries.
  * **ExperimentalAgent**: Prototypes novel research design workflows.
  * **ValidatorAgent**: Ensures queries remain within CS/Data Science domain.
* **Guardrails**: Input validation and output sanitization to enforce domain constraints and content safety.
* **LLM Handler**: Unified interface to Groq, OpenAI, and Gemini with retry & failover support.
* **Session Memory**: Persistent conversation history and session summaries.

## Installation

```bash
git clone https://github.com/mohitbhoir789/ai_research_lab.git
cd ai_research_lab/backend/app
pip install -e .  # installs backend package
cd ../../frontend
pip install -r ../requirements.txt
```

## Configuration

Create a `.env` in the project root with your API keys:

```env
OPENAI_API_KEY=<your-openai-key>
GROQ_API_KEY=<your-groq-key>
GEMINI_API_KEY=<your-gemini-key>
PINECONE_API_KEY=<your-pinecone-key>
```

Enable the **Generative Language API** in your Google Cloud Console and ensure your `GEMINI_API_KEY` has proper permissions.

## Usage

Run the Streamlit app:

```bash
cd frontend
streamlit run gui.py
```

* **Chats**: Create, rename, delete sessions in the sidebar.
* **Model Settings**: Choose LLM provider (groq/openai/gemini) and model name.
* **Ask anything**: Input a query; guardrails will enforce CS/Data Science domain.
* **Flows**: The app detects intent—learning vs. research vs. direct query vs. experimental—and orchestrates the appropriate agent pipeline.

## Project Structure

```
ai_research_lab/
├── backend/app/
│   ├── agents/            # Agent implementations
│   ├── mcp/               # MCPServer & transport
│   ├── utils/             # LLM handler, embeddings, pdf parser, guardrails, errors
│   └── schemas/           # Pydantic models
├── frontend/              # Streamlit UI (gui.py)
├── memory_store/          # Pinecone namespace files
├── sessions/              # Saved session dumps
├── uploads/               # Uploaded files metadata
├── requirements.txt       # Frontend dependencies
└── setup.py               # Backend package setup
```

## Development

* **Add Agents**: Create new agent classes under `backend/app/agents`, inheriting from `LLMAgent`. Register in `MCPServer._initialize_agents()` and the orchestrator graph.
* **Guardrails**: Update `utils/guardrails.py` to modify allowed domains or blacklist.
* **Fine-Tuning**: Integrate PEFT workflows by adding a `FinetunerAgent` and utility scripts (see Hugging Face PEFT docs).
* **LangGraph**: Optionally orchestrate multi-agent flows via LangGraph for advanced pipelines.

## Contribution

Contributions welcome! Please fork the repo and open a PR with bug fixes or new features.

## License

MIT License
