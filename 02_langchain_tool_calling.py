"""
------------------------------------------------------------
Tutorial 2: Tool Calling with LangChain + Ollama
Author: Dr. Sailesh Conjeti
Course: Generative and Agentic AI: Foundations, Frameworks and Applications
------------------------------------------------------------
Purpose:
Show how a model can request and use Python tools, so answers are produced
using external function outputs instead of only model-generated text.

What Students Will Learn:
- How to define tools with the @tool decorator
- How to bind tools to a chat model
- How tool calls are returned, executed, and fed back to the model

Prerequisites:
- Ollama installed and running locally
- Model pulled: llama3.1:8b
- Python environment with project requirements installed

How to Run:
python 02_langchain_tool_calling.py

Expected Behavior / Output:
- Prints the user question
- Shows tool calls requested by the model
- Executes tools and prints the final grounded response

Key Concepts Covered:
- Tool-augmented reasoning
- HumanMessage and ToolMessage flow
- One-round agent-like execution pattern
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage


# ------------------------------------------------------------
# Tool Definitions
# ------------------------------------------------------------
# Tools are normal Python functions with typed arguments and descriptions.
# The model uses these schemas to decide when to call a tool.
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numeric values and return the product."""
    return a * b


@tool
def word_count(text: str) -> int:
    """Count words in a text string by splitting on whitespace."""
    return len(text.split())


# ------------------------------------------------------------
# Tool Registry + Model Setup
# ------------------------------------------------------------
TOOLS = [multiply, word_count]
TOOL_MAP = {tool_.name: tool_ for tool_ in TOOLS}

# bind_tools(...) tells the model which actions it may request.
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
).bind_tools(TOOLS)


def run_agent(question: str) -> None:
    """
    Run one tool-calling cycle:
    1) send user question,
    2) inspect requested tool calls,
    3) execute tools,
    4) send tool results back,
    5) print final answer.
    """
    print("\n=== User Question ===")
    print(question)

    # Conversation state is tracked as a list of messages.
    messages = [HumanMessage(content=question)]

    # First model response may include plain text and/or tool calls.
    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    if not getattr(ai_msg, "tool_calls", None):
        print("\n=== Model Response ===")
        print(ai_msg.content)
        return

    print("\n=== Tool Calls Requested ===")
    for call in ai_msg.tool_calls:
        print(call)

    # Execute each requested tool and append ToolMessage outputs.
    for call in ai_msg.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        result = TOOL_MAP[tool_name].invoke(tool_args)

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=call["id"],
        )
        messages.append(tool_message)

    # Final model response now has access to tool observations.
    final_response = llm.invoke(messages)

    print("\n=== Final Answer After Tool Use ===")
    print(final_response.content)


# ------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    run_agent("What is 17 multiplied by 23, and how many words are in 'agentic AI is powerful'?")
