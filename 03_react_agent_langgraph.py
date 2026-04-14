"""
------------------------------------------------------------
Tutorial 3: ReAct-Style Agent with LangGraph
Author: Dr. Sailesh Conjeti
Course: Generative and Agentic AI: Foundations, Frameworks and Applications
------------------------------------------------------------
Purpose:
Demonstrate a graph-based ReAct loop where the model can repeatedly call
tools until it has enough information to finish.

What Students Will Learn:
- How to represent conversation as explicit graph state
- How to define model and tool nodes
- How conditional routing creates iterative agent behavior

Prerequisites:
- Ollama installed and running locally
- Model pulled: llama3.1:8b
- Python environment with project requirements installed

How to Run:
python 03_react_agent_langgraph.py

Expected Behavior / Output:
- Runs a question through a LangGraph loop
- Prints the full conversation trace (human/model/tool messages)
- Stops when no further tool calls are requested

Key Concepts Covered:
- ReAct pattern (reason -> act -> observe -> repeat)
- StateGraph nodes, edges, and conditional routing
- Tool execution inside graph orchestration
"""

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ------------------------------------------------------------
# Tool Definitions
# ------------------------------------------------------------
@tool
def get_capital(country: str) -> str:
    """Return a capital city from a small built-in dictionary."""
    capitals = {
        "germany": "Berlin",
        "france": "Paris",
        "india": "New Delhi",
        "spain": "Madrid",
    }
    return capitals.get(country.strip().lower(), "I do not know that country.")


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


# ------------------------------------------------------------
# Tool Registry and Shared Agent State
# ------------------------------------------------------------
TOOLS = [get_capital, add_numbers]
TOOL_MAP = {tool_.name: tool_ for tool_ in TOOLS}


class AgentState(TypedDict):
    """Graph state containing the running conversation history."""
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
).bind_tools(TOOLS)


# ------------------------------------------------------------
# Graph Node Functions
# ------------------------------------------------------------
def call_model(state: AgentState) -> AgentState:
    """Call the LLM with current messages and append its next response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState) -> AgentState:
    """Execute all tool calls requested by the most recent model message."""
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
    """Route to tools when tool calls exist; otherwise end the graph run."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ------------------------------------------------------------
# Graph Construction
# ------------------------------------------------------------
# Structure: START -> model -> (tools or END), and tools -> model.
# This loop is the core ReAct behavior in this tutorial.
graph_builder = StateGraph(AgentState)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", call_tools)

graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "model")

graph = graph_builder.compile()


# ------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------
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
