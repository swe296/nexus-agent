"""LangGraph node implementations for NexusAgent."""

from typing import Dict, Any, List
from src.state import AgentState
from src.tools import call_llm, search_web, SimpleVectorStore

class AgentNodes:
    """Encapsulates the discrete reasoning steps and decision nodes of NexusAgent."""

    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def route_query(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Classifies query intent and chooses initial retrieval route."""
        question = state.get("question", "")
        trace = list(state.get("trace", []))

        system_prompt = (
            "You are a Query Router. Classify the user question into one of three routes:\n"
            "- 'vectorstore': technical, analytical, or document-based queries.\n"
            "- 'web_search': questions asking about current events, breaking news, or real-time web facts.\n"
            "- 'direct': greetings, self-introductions, or general conversational remarks.\n"
            "Respond ONLY with one word: 'vectorstore', 'web_search', or 'direct'."
        )
        response = call_llm(prompt=question, system_prompt=system_prompt).lower().strip()

        if "web" in response:
            route = "web_search"
        elif "direct" in response:
            route = "direct"
        else:
            route = "vectorstore"

        trace.append({
            "node": "Route Query",
            "action": f"Classified query intent to route: [{route.upper()}]",
            "status": "completed"
        })

        return {"route": route, "trace": trace, "retry_count": 0}

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Retrieves top candidate chunks from vector database."""
        question = state.get("question", "")
        trace = list(state.get("trace", []))

        docs = self.vector_store.similarity_search(question, k=4)
        trace.append({
            "node": "Vector Retrieval",
            "action": f"Retrieved {len(docs)} candidate chunks from vector store.",
            "status": "completed"
        })

        return {"documents": docs, "trace": trace}

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Evaluates retrieved documents for query relevance (Self-RAG filter)."""
        question = state.get("question", "")
        docs = state.get("documents", [])
        trace = list(state.get("trace", []))

        filtered_docs: List[Dict[str, Any]] = []
        web_fallback = False

        if not docs:
            web_fallback = True
            trace.append({
                "node": "Document Grader",
                "action": "Vector store returned 0 documents. Triggering Web Search Fallback.",
                "status": "fallback_triggered"
            })
            return {"documents": [], "web_fallback": True, "trace": trace}

        system_prompt = (
            "You are a strict relevance grader. Assess if the document snippet is relevant to the question.\n"
            "Answer ONLY 'yes' or 'no'."
        )

        for d in docs:
            prompt = f"Question: {question}\nDocument: {d.get('content', '')}"
            res = call_llm(prompt=prompt, system_prompt=system_prompt).lower()
            if "yes" in res:
                filtered_docs.append(d)

        # If less than 50% of candidate docs are relevant or 0 pass, trigger fallback
        if len(filtered_docs) == 0:
            web_fallback = True
            trace.append({
                "node": "Document Grader",
                "action": f"Filtered 0/{len(docs)} documents as relevant. Triggering Web Search Fallback.",
                "status": "fallback_triggered"
            })
        else:
            trace.append({
                "node": "Document Grader",
                "action": f"Validated {len(filtered_docs)}/{len(docs)} chunks as highly relevant.",
                "status": "completed"
            })

        return {
            "documents": filtered_docs if filtered_docs else docs,
            "web_fallback": web_fallback,
            "trace": trace
        }

    def web_search(self, state: AgentState) -> Dict[str, Any]:
        """Node 4: Fetches real-time web intelligence when local context is insufficient."""
        question = state.get("question", "")
        existing_docs = list(state.get("documents", []))
        trace = list(state.get("trace", []))

        web_results = search_web(question, max_results=3)
        combined_docs = existing_docs + web_results

        trace.append({
            "node": "Web Search Fallback",
            "action": f"Retrieved {len(web_results)} live web sources to supplement local knowledge base.",
            "status": "completed"
        })

        return {"documents": combined_docs, "web_fallback": False, "trace": trace}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Node 5: Synthesizes a factual, grounded answer strictly from validated sources."""
        question = state.get("question", "")
        docs = state.get("documents", [])
        trace = list(state.get("trace", []))
        route = state.get("route", "")

        if route == "direct" and not docs:
            system_prompt = "You are NexusAgent, an intelligent autonomous research AI assistant. Provide a helpful, concise response."
            generation = call_llm(prompt=question, system_prompt=system_prompt)
            trace.append({
                "node": "Direct Synthesizer",
                "action": "Generated conversational response directly without document grounding.",
                "status": "completed"
            })
            return {"generation": generation, "trace": trace, "is_grounded": True}

        context = "\n---\n".join([d.get("content", "") for d in docs])
        system_prompt = (
            "You are NexusAgent, an advanced AI research assistant. Synthesize a clear, accurate, and comprehensive answer "
            "based strictly on the provided Context. If facts are absent from the context, acknowledge the limitation rather than guessing.\n\n"
            f"Context:\n{context}"
        )
        generation = call_llm(prompt=question, system_prompt=system_prompt)

        trace.append({
            "node": "Answer Synthesizer",
            "action": f"Generated grounded response across {len(docs)} context references.",
            "status": "completed"
        })

        return {"generation": generation, "trace": trace}

    def check_hallucination(self, state: AgentState) -> Dict[str, Any]:
        """Node 6: Self-Correction loop evaluating groundedness and factual fidelity."""
        docs = state.get("documents", [])
        generation = state.get("generation", "")
        trace = list(state.get("trace", []))
        retry_count = state.get("retry_count", 0)

        if not docs:
            return {"is_grounded": True, "trace": trace}

        context = "\n".join([d.get("content", "")[:250] for d in docs])
        system_prompt = (
            "You are a Hallucination Evaluator. Verify if the candidate response is strictly grounded in the Context facts.\n"
            "Answer ONLY 'faithful' if all claims are grounded, or 'hallucination' if ungrounded claims exist."
        )
        prompt = f"Context:\n{context}\n\nCandidate Response:\n{generation}"
        result = call_llm(prompt=prompt, system_prompt=system_prompt).lower()

        is_grounded = "faithful" in result or "yes" in result
        trace.append({
            "node": "Hallucination Grader",
            "action": f"Groundedness verification: {'PASSED (Faithful)' if is_grounded else 'FAILED (Ungrounded detected)'}",
            "status": "passed" if is_grounded else "retry_required"
        })

        return {
            "is_grounded": is_grounded,
            "retry_count": retry_count + (1 if not is_grounded else 0),
            "trace": trace
        }
