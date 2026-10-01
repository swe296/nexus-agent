"""Streamlit Web Interface for NexusAgent: Self-Corrective Multi-Agent Research System."""

import streamlit as st
from src.graph import NexusGraph
from src.tools import SimpleVectorStore
from src.config import GEMINI_API_KEY, TAVILY_API_KEY

st.set_page_config(
    page_title="NexusAgent | Self-Corrective Multi-Agent RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .status-badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
    .badge-success { background-color: #DEF7EC; color: #03543F; }
    .badge-info { background-color: #E1EFFE; color: #1E429F; }
    .trace-card { background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px; font-family: monospace; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "vector_store" not in st.session_state:
    vs = SimpleVectorStore()
    # Default baseline documents
    default_docs = [
        "NexusAgent implements Self-Corrective RAG using cyclic LangGraph state machines.",
        "Traditional RAG pipelines fail when vector similarity retrieves noisy or irrelevant contexts.",
        "Document Graders filter retrieved chunks before LLM synthesis to avoid context pollution.",
        "Web Search Fallback triggers dynamically when local knowledge base lacks sufficient facts.",
        "Hallucination Graders evaluate candidate responses against retrieved facts to ensure groundedness."
    ]
    vs.add_texts(default_docs)
    st.session_state.vector_store = vs

if "graph" not in st.session_state:
    st.session_state.graph = NexusGraph(vector_store=st.session_state.vector_store)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("NexusAgent Control")
    st.markdown("Autonomous Self-Corrective Multi-Agent RAG system built with **LangGraph**.")
    
    st.divider()
    st.subheader("⚙️ System Status")
    
    if GEMINI_API_KEY:
        st.markdown('<span class="status-badge badge-success">● Gemini API Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-info">● Demo Engine Active</span>', unsafe_allow_html=True)
        st.caption("Add `GEMINI_API_KEY` to `.env` for live LLM inference.")

    if TAVILY_API_KEY:
        st.markdown('<span class="status-badge badge-success">● Tavily Search Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-info">● DuckDuckGo Search Fallback</span>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📚 Knowledge Base")
    doc_count = len(st.session_state.vector_store.documents)
    st.info(f"Indexed Chunks: **{doc_count}**")

    # Document Uploader
    uploaded_files = st.file_uploader("Upload Documents (.txt, .md)", type=["txt", "md"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            text = f.read().decode("utf-8")
            chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
            if chunks:
                st.session_state.vector_store.add_texts(chunks, metadatas=[{"source": f.name} for _ in chunks])
        st.success(f"Indexed uploaded content! Total chunks: {len(st.session_state.vector_store.documents)}")

    if st.button("Reset Knowledge Base"):
        st.session_state.vector_store = SimpleVectorStore()
        st.session_state.graph = NexusGraph(vector_store=st.session_state.vector_store)
        st.rerun()

# Main Interface
st.markdown('<div class="main-title">NexusAgent: Self-Corrective RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Agent State Graph Orchestration with Automated Relevance Grading & Groundedness Checks</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("💡 Why does naive RAG hallucinate?"):
        st.session_state.active_query = "Why does naive RAG hallucinate?"
with col_b:
    if st.button("🔍 Latest AI agent breakthroughs 2026"):
        st.session_state.active_query = "Latest AI agent breakthroughs 2026"
with col_c:
    if st.button("⚡ How does Document Grading work?"):
        st.session_state.active_query = "How does Document Grading work?"

# User Input
query = st.chat_input("Enter your research question...")
if not query and "active_query" in st.session_state:
    query = st.session_state.pop("active_query")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Agents coordinating reasoning graph..."):
        result = st.session_state.graph.run(query)

    response_payload = {
        "role": "assistant",
        "content": result.get("generation", "No response generated."),
        "trace": result.get("trace", []),
        "docs": result.get("documents", [])
    }
    st.session_state.chat_history.append(response_payload)

# Render Chat History
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

            # Trace Accordion
            trace = message.get("trace", [])
            if trace:
                with st.expander("🔍 Multi-Agent Execution Trace & State Transitions", expanded=False):
                    for step in trace:
                        status = step.get("status", "ok")
                        icon = "✅" if status in ["completed", "passed"] else "⚠️"
                        st.markdown(
                            f'<div class="trace-card">{icon} <b>[{step.get("node")}]</b>: {step.get("action")}</div>',
                            unsafe_allow_html=True
                        )

            # References Accordion
            docs = message.get("docs", [])
            if docs:
                with st.expander(f"📖 Context References & Grounded Sources ({len(docs)})", expanded=False):
                    for i, d in enumerate(docs, 1):
                        source_info = d.get("url") or d.get("metadata", {}).get("source") or "Vector Memory"
                        st.markdown(f"**Source {i}** *({source_info})*:")
                        st.text(d.get("content", "")[:350] + "...")
