"""
------------------------------------------------------------
Tutorial 2: Tool Calling with LangChain + Ollama
Author: Dr. Sailesh Conjeti
------------------------------------------------------------
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

@tool
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())

TOOLS = [multiply, word_count]
TOOL_MAP = {tool_.name: tool_ for tool_ in TOOLS}

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
).bind_tools(TOOLS)

def run_agent(question: str) -> None:
    print("\n=== User Question ===")
    print(question)

    messages = [HumanMessage(content=question)]

    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    if not getattr(ai_msg, "tool_calls", None):
        print("\n=== Model Response ===")
        print(ai_msg.content)
        return

    print("\n=== Tool Calls Requested ===")
    for call in ai_msg.tool_calls:
        print(call)

    for call in ai_msg.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        result = TOOL_MAP[tool_name].invoke(tool_args)

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=call["id"],
        )
        messages.append(tool_message)

    final_response = llm.invoke(messages)

    print("\n=== Final Answer After Tool Use ===")
    print(final_response.content)

if __name__ == "__main__":
    run_agent("What is 17 multiplied by 23, and how many words are in 'agentic AI is powerful'?")