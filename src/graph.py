"""LangGraph workflow definition for the legal assistant."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    drafting_node,
    feedback_node,
    ingest_node,
    rag_chain_node,
    simulation_node,
    summarize_node,
)
from .state import AssistantState

workflow = StateGraph(AssistantState)
workflow.add_node("ingest", ingest_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("rag", rag_chain_node)
workflow.add_node("draft", drafting_node)
workflow.add_node("simulate", simulation_node)
workflow.add_node("feedback", feedback_node)

workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "summarize")
workflow.add_edge("summarize", "rag")
workflow.add_edge("rag", "draft")
workflow.add_edge("draft", "simulate")
workflow.add_edge("simulate", "feedback")
workflow.add_edge("feedback", END)

app = workflow.compile()
