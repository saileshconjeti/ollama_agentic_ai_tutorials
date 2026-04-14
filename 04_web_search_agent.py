"""
------------------------------------------------------------
Tutorial 4: Web Search Agent with a Local Model
Author: Dr. Sailesh Conjeti
Course: Generative and Agentic AI: Foundations, Frameworks and Applications
------------------------------------------------------------
Purpose:
Show how a local model can use an external web-search tool to answer
questions with fresher, more grounded information.

What Students Will Learn:
- How to wrap search as a LangChain tool
- How the model requests tool calls at runtime
- How tool outputs are returned back to the model for final synthesis

Prerequisites:
- Ollama installed and running locally
- Model pulled: llama3.1:8b
- Python environment with project requirements installed
- Internet access (required for web search)

How to Run:
python 04_web_search_agent.py

Expected Behavior / Output:
- Prints initial model message
- Shows requested tool calls
- Displays search summaries and final answer

Key Concepts Covered:
- Tool-augmented answering with current information
- Model/tool handoff and result feedback loop
- Practical grounding beyond model-only memory
"""

from ddgs import DDGS
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama


# ------------------------------------------------------------
# Tool Definition
# ------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """
    Search the web and return a readable text summary of top results.

    The returned string is passed back to the model as tool output,
    allowing the model to build a final response from retrieved evidence.
    """
    results_text = []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No search results found."

    for i, item in enumerate(results, start=1):
        title = item.get("title", "")
        body = item.get("body", "") or item.get("snippet", "")
        href = item.get("href", "") or item.get("link", "")

        results_text.append(
            f"{i}. {title}\n{body}\nURL: {href}"
        )

    return "\n\n".join(results_text)


# ------------------------------------------------------------
# Tool Registry + Model Setup
# ------------------------------------------------------------
TOOLS = [web_search]
TOOL_MAP = {tool_.name: tool_ for tool_ in TOOLS}

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
).bind_tools(TOOLS)


# ------------------------------------------------------------
# Agent Runner
# ------------------------------------------------------------
def answer_question(question: str) -> None:
    """
    Run one question through model -> tool (optional) -> model flow.

    This is a single-round tool-calling pattern, not a looping agent graph.
    """
    messages = [HumanMessage(content=question)]
    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    print("\n=== Initial Model Message ===")
    print(ai_msg.content)
    if getattr(ai_msg, "tool_calls", None):
        print("\n=== Tool Calls ===")
        for call in ai_msg.tool_calls:
            print(call)

    if not getattr(ai_msg, "tool_calls", None):
        print("\n=== Final Answer ===")
        print(ai_msg.content)
        return

    # Execute requested tools and return observations to the model.
    for call in ai_msg.tool_calls:
        result = TOOL_MAP[call["name"]].invoke(call["args"])
        print("\n=== Tool Result ===")
        print(result[:3000])
        messages.append(
            ToolMessage(content=result, tool_call_id=call["id"])
        )

    final_answer = llm.invoke(messages)
    print("\n=== Final Answer ===")
    print(final_answer.content)


# ------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    question = "Find recent developments in agentic AI security and summarize them for students."
    answer_question(question)
