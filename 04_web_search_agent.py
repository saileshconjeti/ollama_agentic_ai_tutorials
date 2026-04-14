"""
------------------------------------------------------------
Tutorial 4: Web Search Agent with a Local Model
Author: Dr. Sailesh Conjeti
------------------------------------------------------------
"""

from ddgs import DDGS
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama


@tool
def web_search(query: str) -> str:
    """Search the web and return top result summaries."""
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


TOOLS = [web_search]
TOOL_MAP = {tool_.name: tool_ for tool_ in TOOLS}

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
).bind_tools(TOOLS)


def answer_question(question: str) -> None:
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


if __name__ == "__main__":
    question = "Find recent developments in agentic AI security and summarize them for students."
    answer_question(question)