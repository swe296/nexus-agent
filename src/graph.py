"""Workflow Graph Construction for NexusAgent."""

from typing import Dict, Any, Optional
from src.state import AgentState
from src.tools import SimpleVectorStore
from src.nodes import AgentNodes

class NexusGraph:
    """Orchestrates the Self-Corrective Multi-Agent RAG workflow."""

    def __init__(self, vector_store: Optional[SimpleVectorStore] = None):
        self.vector_store = vector_store or SimpleVectorStore()
        self.nodes = AgentNodes(vector_store=self.vector_store)
        self._compiled_graph = None
        self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END

            workflow = StateGraph(AgentState)

            # Register Nodes
            workflow.add_node("route_query", self.nodes.route_query)
            workflow.add_node("retrieve", self.nodes.retrieve)
            workflow.add_node("grade_documents", self.nodes.grade_documents)
            workflow.add_node("web_search", self.nodes.web_search)
            workflow.add_node("generate", self.nodes.generate)
            workflow.add_node("check_hallucination", self.nodes.check_hallucination)

            # Entry Point
            workflow.set_entry_point("route_query")

            # Route conditional edges
            workflow.add_conditional_edges(
                "route_query",
                self._route_decision,
                {
                    "direct": "generate",
                    "web_search": "web_search",
                    "vectorstore": "retrieve"
                }
            )

            workflow.add_edge("retrieve", "grade_documents")

            # Document Grading conditional edges
            workflow.add_conditional_edges(
                "grade_documents",
                lambda state: "web_search" if state.get("web_fallback") else "generate",
                {
                    "web_search": "web_search",
                    "generate": "generate"
                }
            )

            workflow.add_edge("web_search", "generate")
            workflow.add_edge("generate", "check_hallucination")

            # Hallucination check loop / termination
            workflow.add_conditional_edges(
                "check_hallucination",
                self._hallucination_decision,
                {
                    "end": END,
                    "retry": "generate"
                }
            )

            self._compiled_graph = workflow.compile()
        except ImportError:
            # Native fallback engine if langgraph is not installed
            self._compiled_graph = None

    def _route_decision(self, state: AgentState) -> str:
        route = state.get("route", "vectorstore")
        return route if route in ["direct", "web_search", "vectorstore"] else "vectorstore"

    def _hallucination_decision(self, state: AgentState) -> str:
        if state.get("is_grounded", True) or state.get("retry_count", 0) >= 2:
            return "end"
        return "retry"

    def run(self, question: str) -> Dict[str, Any]:
        """Execute the workflow for a given question and return final state with full trace."""
        initial_state: AgentState = {
            "question": question,
            "documents": [],
            "trace": [],
            "retry_count": 0,
            "web_fallback": False
        }

        # If compiled langgraph is available, invoke it
        if self._compiled_graph is not None:
            return self._compiled_graph.invoke(initial_state)

        # Fallback pure state-machine execution
        state = dict(initial_state)
        state.update(self.nodes.route_query(state))
        route = self._route_decision(state)

        if route == "direct":
            state.update(self.nodes.generate(state))
        elif route == "web_search":
            state.update(self.nodes.web_search(state))
            state.update(self.nodes.generate(state))
            state.update(self.nodes.check_hallucination(state))
        else:
            state.update(self.nodes.retrieve(state))
            state.update(self.nodes.grade_documents(state))
            if state.get("web_fallback"):
                state.update(self.nodes.web_search(state))
            state.update(self.nodes.generate(state))
            state.update(self.nodes.check_hallucination(state))
            if not state.get("is_grounded") and state.get("retry_count", 0) < 2:
                state.update(self.nodes.generate(state))
                state.update(self.nodes.check_hallucination(state))

        return state
