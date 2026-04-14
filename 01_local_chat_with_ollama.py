"""
------------------------------------------------------------
Tutorial 1: Local Chat with Ollama + LangChain
Author: Dr. Sailesh Conjeti
------------------------------------------------------------
"""

from langchain_ollama import ChatOllama

# Create the local model client
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.2,
)

def single_prompt_demo() -> None:
    print("\n=== Single Prompt Demo ===")
    response = llm.invoke("Explain in simple terms what an AI agent is.")
    print(response.content)

def interactive_chat() -> None:
    print("\n=== Interactive Chat ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = llm.invoke(user_input)
        print(f"Assistant: {response.content}\n")

if __name__ == "__main__":
    single_prompt_demo()
    interactive_chat()