# Ollama Agentic AI Tutorials

Step-by-step local tutorial series for teaching agentic AI with Ollama, LangChain, LangGraph, and Streamlit.

Author: Dr. Sailesh Conjeti  
Course: Generative and Agentic AI: Foundations, Frameworks and Applications

## 0) Learning Goals

By the end of this tutorial series, students should understand:

- what an LLM app is
- what makes a system agentic
- when to use tools instead of raw prompting
- how ReAct loops work
- how workflows differ from free-form agents
- what RAG is and why it improves grounded answers
- how to run an end-to-end local AI stack with Ollama

## 1) Repository Contents

- `01_local_chat_with_ollama.py` - basic local chat
- `02_langchain_tool_calling.py` - one-round tool-calling pattern
- `03_react_agent_langgraph.py` - ReAct-style graph loop
- `04_web_search_agent.py` - tool-based web lookup with a local model
- `05_stateful_workflow_langgraph.py` - structured multi-step workflow
- `06_pdf_rag_chatbot_streamlit.py` - PDF RAG chatbot UI
- `test_streamlit.py` - minimal Streamlit sanity check
- `requirements.txt` - project dependencies

## 2) Setup (Copy-Paste)

### 2.1 Clone

```bash
git clone https://github.com/saileshconjeti/ollama_agentic_ai_tutorials.git
cd ollama_agentic_ai_tutorials
```

### 2.2 Create virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2.3 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2.4 Install and verify Ollama

```bash
ollama --version
ollama list
```

If Ollama is missing on macOS:

```bash
brew install ollama
brew services start ollama
```

### 2.5 Pull models used by this project

```bash
ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull qwen3-embedding:0.6b
```

## 3) Tutorial 1: Local Chat with Ollama

Purpose:
- Build the simplest local LLM app and introduce the prompt-response loop.

What Students Will Learn:
- How to initialize `ChatOllama`
- How to run one fixed prompt
- How to run a terminal chat loop

Prerequisites:
- Ollama running locally
- Model pulled: `qwen3:4b`

How to Run:

```bash
python 01_local_chat_with_ollama.py
```

Expected Behavior / Output:
- Prints one sample answer
- Starts interactive chat until `exit` or `quit`

Key Concepts Covered:
- Local inference
- Model configuration (`temperature`)
- Stateless interaction

## 4) Tutorial 2: Tool Calling with LangChain

Purpose:
- Show how an LLM can call Python tools and use tool results in final answers.

What Students Will Learn:
- How to define tools with `@tool`
- How to bind tools to a model
- How `HumanMessage` and `ToolMessage` are used

Prerequisites:
- Ollama running locally
- Model pulled: `llama3.1:8b`

How to Run:

```bash
python 02_langchain_tool_calling.py
```

Expected Behavior / Output:
- Prints user question
- Shows requested tool calls
- Executes tools and prints grounded final answer

Key Concepts Covered:
- Tool-augmented reasoning
- Model -> tool -> model flow
- One-round agent-like execution

## 5) Tutorial 3: ReAct-Style Agent with LangGraph

Purpose:
- Demonstrate a ReAct loop where the model can repeatedly use tools until done.

What Students Will Learn:
- How to represent state in LangGraph
- How to define `model` and `tools` nodes
- How conditional routing controls loop behavior

Prerequisites:
- Ollama running locally
- Model pulled: `llama3.1:8b`

How to Run:

```bash
python 03_react_agent_langgraph.py
```

Expected Behavior / Output:
- Runs through a graph loop
- Prints full conversation trace (human/model/tool messages)
- Ends when no tool calls remain

Key Concepts Covered:
- ReAct pattern (`reason -> act -> observe -> repeat`)
- Graph nodes, edges, routing, and loops
- Tool execution inside orchestration

## 6) Tutorial 4: Web Search Agent

Purpose:
- Show how a local model can use a web-search tool for fresher answers.

What Students Will Learn:
- How to wrap search with a LangChain tool
- How model-requested tool calls execute at runtime
- How search results are fed back for synthesis

Prerequisites:
- Ollama running locally
- Model pulled: `llama3.1:8b`
- Internet access

How to Run:

```bash
python 04_web_search_agent.py
```

Expected Behavior / Output:
- Prints initial model message
- Shows tool calls
- Displays tool results and final answer

Key Concepts Covered:
- Tool grounding for current information
- Model/tool handoff
- Practical mitigation of stale model-only answers

## 7) Tutorial 5: Stateful Workflow with LangGraph

Purpose:
- Demonstrate a fixed, deterministic AI workflow with shared typed state.

What Students Will Learn:
- How to define workflow state
- How to build sequential nodes
- How state evolves from plan to final output

Prerequisites:
- Ollama running locally
- Model pulled: `qwen3:4b`

How to Run:

```bash
python 05_stateful_workflow_langgraph.py
```

Expected Behavior / Output:
- Executes the full pipeline
- Prints the final improved draft

Key Concepts Covered:
- Stateful graph orchestration
- Predictable workflow design
- Workflow vs open-ended agent design

## 8) Tutorial 6: PDF RAG Chatbot (Streamlit)

Purpose:
- Build a local RAG app where students upload PDFs and ask grounded questions.

What Students Will Learn:
- PDF loading and parsing
- Chunking for retrieval
- Embeddings + vector store indexing
- Retrieval + grounded answer generation in a UI

Prerequisites:
- Ollama running locally
- Chat models available: `qwen3:4b`, `gemma3:4b`, `llama3.1:8b`
- Embedding model available: `nomic-embed-text` or `qwen3-embedding:0.6b`

How to Run:

```bash
streamlit run 06_pdf_rag_chatbot_streamlit.py
```

Expected Behavior / Output:
- Upload PDFs
- Build index from chunks
- Ask questions and receive context-grounded answers
- View retrieved chunks in the UI

Key Concepts Covered:
- Retrieval-Augmented Generation (RAG)
- Document loading, chunking, embeddings, vector search
- Grounded generation with retrieved context

Important:
- Run Streamlit files with `streamlit run ...`, not `python ...`

## 9) Tutorial Utility: Streamlit Environment Sanity Check

Purpose:
- Verify Streamlit setup independently before debugging larger demos.

How to Run:

```bash
streamlit run test_streamlit.py
```

Expected Behavior / Output:
- Browser opens and displays a simple success message.

## 10) Suggested Teaching Sequence

1. `python 01_local_chat_with_ollama.py`
2. `python 02_langchain_tool_calling.py`
3. `python 03_react_agent_langgraph.py`
4. `python 04_web_search_agent.py`
5. `python 05_stateful_workflow_langgraph.py`
6. `streamlit run 06_pdf_rag_chatbot_streamlit.py`

## 11) Troubleshooting

### `ModuleNotFoundError` (streamlit/langchain/etc.)

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama connection errors (`127.0.0.1:11434`)

```bash
ollama list
```

If needed on macOS:

```bash
brew services start ollama
```

### Model not found

```bash
ollama pull qwen3:4b
```

Use the model name shown in the error if different.

### Tool calling appears inconsistent

Use:

- `llama3.1:8b`

### PDF RAG feels slow

Try:

- smaller PDFs
- lower `chunk_size`
- lower `top_k`
- `nomic-embed-text`
- lighter chat model (`qwen3:4b` or `gemma3:4b`)

### Streamlit warns about watchdog performance

Optional:

```bash
pip install watchdog
```

## 12) License / Usage

Use and adapt freely for educational teaching.
