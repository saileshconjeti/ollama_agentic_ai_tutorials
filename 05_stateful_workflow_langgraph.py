"""
------------------------------------------------------------
Tutorial 5: Stateful Workflow with LangGraph
Author: Dr. Sailesh Conjeti
------------------------------------------------------------
"""

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

class WorkflowState(TypedDict, total=False):
    topic: str
    plan: str
    research_notes: str
    draft: str
    critique: str
    final_output: str

llm = ChatOllama(model="qwen3:4b", temperature=0.2)

def plan_node(state: WorkflowState) -> WorkflowState:
    prompt = f"Create a short lesson plan for teaching this topic: {state['topic']}"
    response = llm.invoke(prompt)
    return {"plan": response.content}

def research_node(state: WorkflowState) -> WorkflowState:
    prompt = (
        f"Given this topic: {state['topic']}\n"
        f"And this lesson plan:\n{state['plan']}\n\n"
        "Generate short research notes with key concepts, examples, and common misconceptions."
    )
    response = llm.invoke(prompt)
    return {"research_notes": response.content}

def draft_node(state: WorkflowState) -> WorkflowState:
    prompt = (
        f"Write a student-facing teaching draft on:\n{state['topic']}\n\n"
        f"Plan:\n{state['plan']}\n\n"
        f"Research notes:\n{state['research_notes']}"
    )
    response = llm.invoke(prompt)
    return {"draft": response.content}

def critique_node(state: WorkflowState) -> WorkflowState:
    prompt = (
        "Critique the following teaching draft. "
        "Focus on clarity, completeness, and whether students may misunderstand anything.\n\n"
        f"{state['draft']}"
    )
    response = llm.invoke(prompt)
    return {"critique": response.content}

def finalize_node(state: WorkflowState) -> WorkflowState:
    prompt = (
        "Improve the teaching draft using the critique below.\n\n"
        f"Draft:\n{state['draft']}\n\n"
        f"Critique:\n{state['critique']}"
    )
    response = llm.invoke(prompt)
    return {"final_output": response.content}

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

if __name__ == "__main__":
    result = graph.invoke({"topic": "ReAct agents versus fixed workflows"})
    print("\n=== Final Output ===\n")
    print(result["final_output"])