"""
------------------------------------------------------------
Tutorial 1: Local Chat with Ollama + LangChain
Author: Dr. Sailesh Conjeti
Course: Generative and Agentic AI: Foundations, Frameworks and Applications
------------------------------------------------------------
Purpose:
Build the simplest possible local LLM application and show a basic
prompt-response loop before introducing tools, workflows, or retrieval.

What Students Will Learn:
- How to initialize a local chat model with Ollama via LangChain
- How to send a single prompt and read model output
- How to run an interactive terminal chat loop

Prerequisites:
- Ollama installed and running locally
- Model pulled: qwen3:4b
- Python environment with project requirements installed

How to Run:
python 01_local_chat_with_ollama.py

Expected Behavior / Output:
- First, prints one sample response for a fixed question
- Then opens an interactive chat loop until user types exit/quit

Key Concepts Covered:
- Local inference with Ollama
- Chat model configuration (model name, temperature)
- Stateless prompt-response interaction
"""

from langchain_ollama import ChatOllama

# ------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------
# This object is the local LLM client. All invocations in this file use it.
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.2,
)


def single_prompt_demo() -> None:
    """Run a single fixed prompt to demonstrate the basic LLM call pattern."""
    print("\n=== Single Prompt Demo ===")
    response = llm.invoke("Explain in simple terms what an AI agent is.")
    print(response.content)


def interactive_chat() -> None:
    """Start a simple terminal chat loop and stop when the user exits."""
    print("\n=== Interactive Chat ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = llm.invoke(user_input)
        print(f"Assistant: {response.content}\n")


# ------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    single_prompt_demo()
    interactive_chat()
