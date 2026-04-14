"""
------------------------------------------------------------
Tutorial 5: Stateful Workflow with LangGraph
Author: Dr. Sailesh Conjeti
Course: Generative and Agentic AI: Foundations, Frameworks and Applications
------------------------------------------------------------
Purpose:
Demonstrate a fixed, multi-step AI workflow where each step writes to
explicit shared state, producing predictable and controllable behavior.

What Students Will Learn:
- How to design a typed workflow state
- How to build sequential LangGraph nodes
- How state moves across planning, research, drafting, critique, and revision

Prerequisites:
- Ollama installed and running locally
- Model pulled: qwen3:4b
- Python environment with project requirements installed

How to Run:
python 05_stateful_workflow_langgraph.py

Expected Behavior / Output:
- Executes a full pipeline from topic to final improved draft
- Prints the final teaching output at the end

Key Concepts Covered:
- Stateful graph orchestration
- Deterministic workflow design
- Difference between fixed workflows and open-ended agents
"""

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# ------------------------------------------------------------
# Shared Workflow State
# ------------------------------------------------------------
# total=False means fields are optional at initialization and get filled in
# as each node writes its output into state.
class WorkflowState(TypedDict, total=False):
    """State container passed between all workflow nodes."""
    topic: str
    plan: str
    research_notes: str
    draft: str
    critique: str
    final_output: str


# ------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------
llm = ChatOllama(model="qwen3:4b", temperature=0.2)


# ------------------------------------------------------------
# Workflow Nodes
# ------------------------------------------------------------
def plan_node(state: WorkflowState) -> WorkflowState:
    """Create a concise lesson plan from the input topic."""
    prompt = f"Create a short lesson plan for teaching this topic: {state['topic']}"
    response = llm.invoke(prompt)
    return {"plan": response.content}


def research_node(state: WorkflowState) -> WorkflowState:
    """Generate targeted research notes based on topic and plan."""
    prompt = (
        f"Given this topic: {state['topic']}\n"
        f"And this lesson plan:\n{state['plan']}\n\n"
        "Generate short research notes with key concepts, examples, and common misconceptions."
    )
    response = llm.invoke(prompt)
    return {"research_notes": response.content}


def draft_node(state: WorkflowState) -> WorkflowState:
    """Produce a student-facing draft using plan and research notes."""
    prompt = (
        f"Write a student-facing teaching draft on:\n{state['topic']}\n\n"
        f"Plan:\n{state['plan']}\n\n"
        f"Research notes:\n{state['research_notes']}"
    )
    response = llm.invoke(prompt)
    return {"draft": response.content}


def critique_node(state: WorkflowState) -> WorkflowState:
    """Critique the draft for clarity and potential student misunderstandings."""
    prompt = (
        "Critique the following teaching draft. "
        "Focus on clarity, completeness, and whether students may misunderstand anything.\n\n"
        f"{state['draft']}"
    )
    response = llm.invoke(prompt)
    return {"critique": response.content}


def finalize_node(state: WorkflowState) -> WorkflowState:
    """Revise the draft using critique feedback to produce final output."""
    prompt = (
        "Improve the teaching draft using the critique below.\n\n"
        f"Draft:\n{state['draft']}\n\n"
        f"Critique:\n{state['critique']}"
    )
    response = llm.invoke(prompt)
    return {"final_output": response.content}


# ------------------------------------------------------------
# Graph Construction
# ------------------------------------------------------------
# This graph is intentionally linear and predictable:
# START -> plan -> research -> draft -> critique -> finalize -> END
builder = StateGraph(WorkflowState)
builder.add_node("plan", plan_node)
builder.add_node("research", research_node)
builder.add_node("draft", draft_node)
builder.add_node("critique", critique_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "plan")
builder.add_edge("plan", "research")
builder.add_edge("research", "draft")
builder.add_edge("draft", "critique")
builder.add_edge("critique", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()


# ------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    result = graph.invoke({"topic": "ReAct agents versus fixed workflows"})
    print("\n=== Final Output ===\n")
    print(result["final_output"])
