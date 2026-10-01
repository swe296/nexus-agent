"""State schema definition for the NexusAgent LangGraph workflow."""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    """Represents the shared memory state passed between LangGraph nodes."""
    question: str
    transformed_query: Optional[str]
    route: str  # "vectorstore" | "web_search" | "direct"
    documents: List[Dict[str, Any]]
    web_fallback: bool
    relevance_scores: List[Dict[str, Any]]
    generation: str
    is_grounded: bool
    hallucination_score: float
    retry_count: int
    trace: List[Dict[str, str]]
    error: Optional[str]
