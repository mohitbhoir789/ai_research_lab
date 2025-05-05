# 🧠 AI Research Lab

An interactive Streamlit-based application for AI-driven research assistance in Computer Science and Data Science. Powered by a modular multi-agent architecture via the Model Context Protocol (MCP).

## Features

* **Streamlit Frontend**: Chat-like UI for user queries, file uploads, and session management.

* **MCPServer Orchestration**: Central orchestrator routing user input and managing specialized agents based on detected intent.

* **Multi-Agent Flow**:

  * **User Input**: Captures the user's question or query.
  * **Intent Detection**: Classifies the input into categories (e.g., retrieval, research).
  * **Retriever Agent**: Fetches relevant context from Pinecone, Wikipedia, or ArXiv.
  * **Researcher Agent**: Gathers metadata and abstracts from ArXiv for research questions.
  * **Verifier Agent**: Validates factual accuracy of retrieved or researched content.
  * **Summarizer Agent**: Generates both bullet-point and paragraph summaries from verified content.
  * **Final Output**: Aggregates summarized, validated content for user display.
  * **Fine-tuner Agent**: Handles PEFT-style workflows for model fine-tuning tasks.
  * **Intent Detector Agent**: Advanced intent classification for precise agent selection.

* **Guardrails**:

  * **Input Validation**: Ensures that the user queries fall strictly within the Computer Science and Data Science domains.
  * **Output Sanitization**: Filters out unsafe, misleading, or irrelevant content before presenting to users.

* **LLM Handler**: Unified API interface supporting Groq, OpenAI, and Gemini with retry and failover mechanisms.

* **Session Memory**: Maintains persistent chat history and session-level summaries.

## Installation

```bash
git clone https://github.com/mohitbhoir789/ai_research_lab.git
cd ai_research_lab
pip install -e .  # installs the package in development mode
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your API credentials:

```env
OPENAI_API_KEY=<your-openai-key>
GROQ_API_KEY=<your-groq-key>
GEMINI_API_KEY=<your-gemini-key>
PINECONE_API_KEY=<your-pinecone-key>
```

Ensure the Generative Language API is enabled in Google Cloud Console for Gemini usage.

## Usage

Start the application:

```bash
cd frontend
python app.py
```

* **Sessions**: Manage chats via sidebar (create, rename, delete).
* **Model Settings**: Choose between Groq, OpenAI, and Gemini models.
* **Query**: Ask your research question within the CS/Data Science domain.
* **Flow**: The system routes input based on intent and invokes the appropriate agent pipeline:

  * Intent Detection Agent → Specialized Agent Selection → Content Processing → Verification → Summarization → User Response

## Enhanced Agent Flow

The updated system now includes a more sophisticated flow:

1. **Intent Detection**: Analyzes user input to determine the type of request
2. **Agent Selection**: Routes to one or more specialized agents:
   - **Retriever Agent**: For knowledge-based queries using vector databases
   - **Researcher Agent**: For academic research with ArXiv integration
   - **Fine-tuner Agent**: For model customization requests
   - **Summarizer Agent**: For content condensation and highlighting
   - **Verifier Agent**: For fact-checking and validation
3. **Memory Management**: Long-term retention of important information
4. **Response Generation**: Coherent, verified answers with citations

## Project Structure

```
ai_research_lab/
├── backend/app/
│   ├── agents/                # Agent implementations
│   │   ├── agent_core.py      # Base agent class
│   │   ├── intent_detector_agent.py  # New intent classifier
│   │   ├── finetuner_agent.py # New fine-tuning agent
│   │   ├── summarizer_agent.py
│   │   ├── verifier_agent.py
│   │   └── ... other agents
│   ├── mcp/                   # MCPServer & protocol
│   ├── utils/                 # LLM handler, embeddings, guardrails, memory
│   └── schemas/               # Pydantic models
├── frontend/                  # Frontend application (app.py)
├── memory_store/              # Memory persistence
├── sessions/                  # Session management
├── tests/                     # Test suite
│   ├── conftest.py            # Test configuration
│   ├── run_tests.py           # Test runner
│   └── unit/                  # Unit tests by component
├── requirements.txt           # Dependencies
└── setup.py                   # Package configuration
```

## Testing

Run the test suite:

```bash
python tests/run_tests.py
```

Or using pytest directly:

```bash
pytest tests/
```

## Development

* **Add New Agents**: Extend `LLMAgent` in `backend/app/agents` and register in `MCPServer._initialize_agents()`.
* **Guardrails**: Update logic in `utils/guardrails.py` for allowed topics, unsafe inputs, or toxic outputs.
* **Fine-Tuning**: Use the `FinetunerAgent` for PEFT-style workflows using Hugging Face tools.
* **Advanced Orchestration**: Optionally integrate LangGraph for branching multi-agent control flows.

## Contribution

Contributions are welcome! Fork the repo and open a PR with your updates or new features.

## License

MIT License
