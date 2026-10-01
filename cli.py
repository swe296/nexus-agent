"""Command-line interface to execute queries through NexusAgent."""

import sys
import json
from src.graph import NexusGraph
from src.tools import SimpleVectorStore

def main():
    print("=" * 65)
    print("  NexusAgent: Self-Corrective Multi-Agent RAG System")
    print("=" * 65)

    vector_store = SimpleVectorStore()
    
    # Pre-populate sample facts if empty
    if len(vector_store.documents) == 0:
        vector_store.add_texts([
            "NexusAgent uses LangGraph to construct cyclic state graphs for autonomous reasoning.",
            "Retrieval Augmented Generation (RAG) combines dense vector search with large language models.",
            "Self-RAG introduces selective retrieval, document grading, and hallucination evaluation loops.",
            "ChromaDB is an open-source AI vector database designed for high-throughput similarity indexing."
        ])
        print(f"[*] Initialized vector store with {len(vector_store.documents)} knowledge base documents.")

    graph = NexusGraph(vector_store=vector_store)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How does Self-RAG prevent hallucination in multi-agent workflows?"
    print(f"\n[?] Query: {query}")
    print("\n--- Executing Multi-Agent Reasoning Graph ---")

    result = graph.run(query)

    print("\n[+] Execution Trace:")
    for step in result.get("trace", []):
        status_tag = f"[{step.get('status', 'OK').upper()}]"
        print(f"  • {step.get('node'):<22} | {status_tag:<18} | {step.get('action')}")

    print("\n" + "=" * 65)
    print("FINAL SYNTHESIZED RESPONSE:")
    print("=" * 65)
    print(result.get("generation", "No generation produced."))
    print("=" * 65)

if __name__ == "__main__":
    main()
