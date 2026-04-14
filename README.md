# Ollama Agentic AI Tutorials

Step-by-step local tutorial series for teaching agentic AI with Ollama, LangChain, LangGraph, and Streamlit.

Author: Dr. Sailesh Conjeti

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

## 2) Prerequisites

- Python 3.10+ (recommended: 3.11+)
- Ollama installed and running
- terminal access
- internet access for initial installs and model pulls
- enough RAM for local models (4B runs lighter than 8B)

Recommended chat models in this repo:

- `qwen3:4b` (lightweight default)
- `gemma3:4b` (lightweight alternative)
- `llama3.1:8b` (stronger for tool behavior)

Recommended embedding models in this repo:

- `nomic-embed-text`
- `qwen3-embedding:0.6b`

## 3) Setup (Copy-Paste)

### 3.1 Clone

```bash
git clone https://github.com/saileshconjeti/ollama_agentic_ai_tutorials.git
cd ollama_agentic_ai_tutorials
```

### 3.2 Create virtual environment

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

### 3.3 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Install and verify Ollama

Check Ollama:

```bash
ollama --version
ollama list
```

If Ollama is missing on macOS:

```bash
brew install ollama
brew services start ollama
```

### 3.5 Pull models used by this project

```bash
ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull qwen3-embedding:0.6b
```

Quick concept for students:

- chat model: generates answers
- embedding model: converts text to vectors for retrieval
- chat and embedding models are often different

## 4) Tutorial 1: Local Chat with Ollama

Run:

```bash
python 01_local_chat_with_ollama.py
```

What this file does:

1. Creates a local `ChatOllama` client (`qwen3:4b`)
2. Runs one prompt (`single_prompt_demo`)
3. Starts a terminal chat loop (`interactive_chat`)

Teach this:

- local inference flow: prompt -> model -> response
- temperature effect (`0.2` here)
- no tools, no retrieval, no orchestration yet

## 5) Tutorial 2: Tool Calling with LangChain

Run:

```bash
python 02_langchain_tool_calling.py
```

What this file does:

1. Defines two tools (`multiply`, `word_count`)
2. Binds tools to model (`llama3.1:8b`)
3. Sends question as messages
4. Executes model-requested tool calls
5. Sends tool outputs back and gets final answer

Teach this flow:

1. user asks
2. model decides whether to call tools
3. tool runs with structured arguments
4. tool result is returned as `ToolMessage`
5. model produces final grounded response

Key concept:

- `LLM + tools` introduces agent-like behavior

## 6) Tutorial 3: ReAct-Style Agent with LangGraph

Run:

```bash
python 03_react_agent_langgraph.py
```

What this file does:

1. Defines tools (`get_capital`, `add_numbers`)
2. Defines state (`messages`) with `add_messages`
3. Builds graph with `model` and `tools` nodes
4. Adds conditional routing after model step
5. Loops `model -> tools -> model` until no tool calls

Graph pattern:

`START -> model -> (tools?) -> model -> ... -> END`

Teach this:

- ReAct loop: reason, act, observe, repeat
- difference from Tutorial 2: looped orchestration instead of one round

## 7) Tutorial 4: Web Search Agent

Run:

```bash
python 04_web_search_agent.py
```

What this file does:

1. Defines `web_search` tool using `ddgs`
2. Lets model decide when to search
3. Executes search tool and returns summaries + URLs
4. Produces final answer after tool observations

Teach this:

- local model can still use external tools
- tools reduce staleness and improve grounding
- model alone may hallucinate or miss recent facts

Note:

- this tutorial depends on internet connectivity for search

## 8) Tutorial 5: Stateful Workflow with LangGraph

Run:

```bash
python 05_stateful_workflow_langgraph.py
```

What this file does:

1. Defines explicit workflow state (`topic`, `plan`, `research_notes`, `draft`, `critique`, `final_output`)
2. Creates fixed nodes: plan -> research -> draft -> critique -> finalize
3. Compiles and runs deterministic graph

Teach this contrast:

- Agent: open-ended, adaptive, tool-driven
- Workflow: structured, predictable, bounded

Design lesson:

- not every intelligent system should be a free-form agent

## 9) Tutorial 6: PDF RAG Chatbot (Streamlit)

Run:

```bash
streamlit run 06_pdf_rag_chatbot_streamlit.py
```

Important:

- use `streamlit run ...`, not `python ...`

What this file does:

1. Uploads one or more PDFs
2. Loads pages with `PyPDFLoader`
3. Splits pages into chunks
4. Embeds chunks with `OllamaEmbeddings`
5. Stores chunks in Chroma vector DB
6. Retrieves top-k relevant chunks per question
7. Prompts chat model with retrieved context
8. Shows answer + retrieved chunks for transparency

Good classroom flow:

1. upload small PDFs
2. click **Build index**
3. ask a question directly answerable from the PDFs
4. inspect retrieved chunks with students

## 10) RAG Explanation for Students

RAG = Retrieval-Augmented Generation.

Pipeline:

1. ingest documents
2. chunk text
3. embed chunks
4. store vectors
5. retrieve relevant chunks for a query
6. pass retrieved context to LLM
7. generate grounded answer

Why RAG helps:

- lowers hallucination risk
- grounds responses in uploaded sources
- enables custom knowledge without fine-tuning

## 11) Sanity Check Script

Run:

```bash
streamlit run test_streamlit.py
```

Use this if you want to confirm Streamlit is healthy before debugging RAG-specific issues.

## 12) Suggested Classroom Teaching Sequence

1. `python 01_local_chat_with_ollama.py` - model basics
2. `python 02_langchain_tool_calling.py` - tool use
3. `python 03_react_agent_langgraph.py` - iterative agent loop
4. `python 04_web_search_agent.py` - external knowledge via tools
5. `python 05_stateful_workflow_langgraph.py` - controlled orchestration
6. `streamlit run 06_pdf_rag_chatbot_streamlit.py` - practical RAG UI demo

Suggested narration:

- start with pure prompting
- add actions through tools
- add iterative loops (ReAct)
- compare against deterministic workflows
- finish with grounded document QA via RAG

## 13) Troubleshooting

### `ModuleNotFoundError` (streamlit/langchain/etc.)

Your venv is likely not active.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama connection errors (`127.0.0.1:11434`)

Ollama is not running or unreachable.

```bash
ollama list
```

If needed on macOS:

```bash
brew services start ollama
```

### Model not found

Pull the missing model:

```bash
ollama pull qwen3:4b
```

(or whichever model the error reports)

### Tool calling looks weak/inconsistent

Use the stronger tool model in these tutorials:

- `llama3.1:8b`

### PDF RAG feels slow

Try:

- `nomic-embed-text`
- smaller PDFs
- lower `chunk_size`
- lower `top_k`
- lighter chat model (`qwen3:4b` / `gemma3:4b`)

### Streamlit warning about watchdog performance

Optional improvement:

```bash
pip install watchdog
```

### Streamlit app not opening automatically

Use URL shown in terminal, usually:

- `http://localhost:8501`

## 14) Good Conceptual Questions for Students

- What changes when a model can take actions, not just generate text?
- Why are tools often more reliable than pure model reasoning for computation/search?
- How is a ReAct loop different from a fixed workflow?
- Why do embeddings matter for RAG?
- Why might chat and embedding models be different?
- When is a workflow safer than a free-form agent?

## 15) One-Shot Setup + Run Summary

```bash
# macOS quick path
brew install ollama
brew services start ollama

ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull qwen3-embedding:0.6b

git clone https://github.com/saileshconjeti/ollama_agentic_ai_tutorials.git
cd ollama_agentic_ai_tutorials

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python 01_local_chat_with_ollama.py
python 02_langchain_tool_calling.py
python 03_react_agent_langgraph.py
python 04_web_search_agent.py
python 05_stateful_workflow_langgraph.py
streamlit run 06_pdf_rag_chatbot_streamlit.py
```

## 16) License / Usage

Use and adapt freely for educational teaching.
