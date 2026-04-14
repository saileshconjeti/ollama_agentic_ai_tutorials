"""
------------------------------------------------------------
Tutorial 3: ReAct-Style Agent with LangGraph
Author: Dr. Sailesh Conjeti
------------------------------------------------------------
"""

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

@tool
def get_capital(country: str) -> str:
    """Return the capital city of a country for a small built-in set."""
    capitals = {
        "germany": "Berlin",
        "france": "Paris",
        "india": "New Delhi",
        "spain": "Madrid",
    }
    return capitals.get(country.strip().lower(), "I do not know that country.")

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

TOOLS = [get_capital, add_numbers]
TOOL_MAP = {tool_.name: tool_ for tool_ in TOOLS}

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
).bind_tools(TOOLS)

def call_model(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_messages = []

    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        result = TOOL_MAP[tool_name].invoke(tool_args)

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )

    return {"messages": tool_messages}

def route_after_model(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

graph_builder = StateGraph(AgentState)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", call_tools)

graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "model")

graph = graph_builder.compile()

if __name__ == "__main__":
    user_question = "What is the capital of Germany, and what is 15 + 27?"

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_question)]}
    )

    print("\n=== Final Conversation ===")
    for msg in result["messages"]:
        msg_type = msg.__class__.__name__
        content = getattr(msg, "content", "")
        print(f"\n[{msg_type}]")
        print(content)