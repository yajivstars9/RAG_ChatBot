import os
import tempfile
import streamlit as st
import logging
import warnings

# Functions - RAG
from rag_ingestion_embedding import *
from rag_llm_pipeline import *
from rag_existing_pipelines import *

# Warnings and logging config
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS", "1")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Accessing `__path__`.*")
warnings.filterwarnings("ignore", message=".*__path__.*")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session initialization ──────────────────────────
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .step-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .step-badge {
        background: #6366f1;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #166534;
    }
    .chat-bubble-user {
        background: #6366f1;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1.1rem;
        max-width: 80%;
        margin-left: auto;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .chat-bubble-bot {
        background: #f1f5f9;
        color: #1e293b;
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1.1rem;
        max-width: 80%;
        margin-right: auto;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .source-tag {
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    div[data-testid="stSidebar"] {
        background: #1e293b;
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stTextInput label {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════
def init_session():
    defaults = {
        "stage": "choose",           # choose → upload → vectorize → chat (or continue → chat)
        "docs": [],
        "chunks": [],
        "retriever": None,
        "vector_store": None,
        "chat_history": [],          # list of {role, content, sources} for display
        "embedding_model": None,
        "reranker_model": None,
        "total_chunks": 0,
        "ingested_files": [],
        # LangGraph state — persisted across turns
        "lg_state": {"messages": [], "summary": "", "sources": []},
        "lg_graph": None,            # compiled graph, rebuilt when retriever is ready
        "query_input": "",
        "clear_query_input": False,
        "pipeline_mode": None,       # "new" or "continue"
        "selected_thread_id": None,  # for continuing from existing
        # Qdrant settings (stored for reuse across pipeline continuations)
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "collection_name": "documents",
        "pg_pool": None,             # PostgreSQL connection pool
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🤖 LLM Backend")
    llm_backend = st.selectbox("Choose LLM", ["Groq", "Ollama (Local)"], key="llm_choice")

    groq_api_key = None
    ollama_model = "mistral"

    if llm_backend == "Groq":
        groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        groq_model = st.selectbox("Groq Model", [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b"

        ])
    else:
        ollama_model = st.selectbox("Ollama Model", ["mistral", "llama3", "phi3"])

    st.markdown("---")
    st.markdown("### 🔧 Embedding Settings")
    embedding_model_name = st.selectbox("Embedding Model", [
        "BAAI/bge-small-en",
        "BAAI/bge-large-en",
        "all-MiniLM-L6-v2"
    ])

    vector_size_map = {
        "BAAI/bge-small-en": 384,
        "BAAI/bge-large-en": 1024,
        "all-MiniLM-L6-v2": 384
    }

    st.markdown("---")
    st.markdown("### 📊 Qdrant Settings")
    qdrant_host = st.text_input("Qdrant Host", value="localhost")
    qdrant_port = st.number_input("Qdrant Port", value=6333, step=1)
    collection_name = st.text_input("Collection Name", value="documents")

    st.markdown("---")
    st.markdown("### 📈 Status")
    if st.session_state.stage == "upload":
        st.info("🔵 Step 1: Upload files")
    elif st.session_state.stage == "vectorize":
        st.warning("🟡 Step 2: Ready to vectorize")
    elif st.session_state.stage == "chat":
        st.success("🟢 Ready to chat!")
        st.metric("Chunks stored", st.session_state.total_chunks)
        st.metric("Files ingested", len(st.session_state.ingested_files))

    if st.session_state.stage == "chat":
        st.markdown("---")
        if st.button("🔄 New Pipeline", use_container_width=True):
            st.session_state.stage = "choose"
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔧 Diagnostics")
    if st.button("🔍 Test Connections", use_container_width=True):
        test_results = []
        
        # Test PostgreSQL
        try:
            from psycopg_pool import ConnectionPool
            pool = ConnectionPool(
                conninfo="postgresql://rag:rag@localhost:5433/RAG_storage",
                min_size=1,
                max_size=5,
                timeout=3
            )
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            test_results.append(("✅ PostgreSQL", "Connected to localhost:5433"))
            pool.close()
        except Exception as e:
            test_results.append(("❌ PostgreSQL", f"Failed: {e}"))
        
        # Test Qdrant
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            info = client.get_collections()
            test_results.append(("✅ Qdrant", f"Connected (found {len(info.collections)} collections)"))
        except Exception as e:
            test_results.append(("❌ Qdrant", f"Failed: {e}"))
        
        for status, msg in test_results:
            if "✅" in status:
                st.success(f"{status}: {msg}")
            else:
                st.error(f"{status}: {msg}")


# ═══════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════
st.markdown('<div class="main-header">🧠 RAG Document Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload documents → Vectorize → Chat with your data using Qdrant + Dense Retrieval + Reranking</div>', unsafe_allow_html=True)

# Progress indicator
col1, col2, col3 = st.columns(3)
with col1:
    color = "#6366f1" if st.session_state.stage == "upload" else ("#10b981" if st.session_state.stage in ["vectorize","chat"] else "#94a3b8")
    st.markdown(f"""<div style="text-align:center; padding:0.5rem; background:{'#ede9fe' if st.session_state.stage=='upload' else '#f0fdf4' if st.session_state.stage in ['vectorize','chat'] else '#f8fafc'}; border-radius:8px; border: 2px solid {color};">
    <b style="color:{color}">① Upload Files</b></div>""", unsafe_allow_html=True)
with col2:
    color = "#6366f1" if st.session_state.stage == "vectorize" else ("#10b981" if st.session_state.stage == "chat" else "#94a3b8")
    st.markdown(f"""<div style="text-align:center; padding:0.5rem; background:{'#ede9fe' if st.session_state.stage=='vectorize' else '#f0fdf4' if st.session_state.stage=='chat' else '#f8fafc'}; border-radius:8px; border: 2px solid {color};">
    <b style="color:{color}">② Vectorize & Store</b></div>""", unsafe_allow_html=True)
with col3:
    color = "#6366f1" if st.session_state.stage == "chat" else "#94a3b8"
    st.markdown(f"""<div style="text-align:center; padding:0.5rem; background:{'#ede9fe' if st.session_state.stage=='chat' else '#f8fafc'}; border-radius:8px; border: 2px solid {color};">
    <b style="color:{color}">③ Chat with Docs</b></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────
# STAGE 0: CHOOSE - NEW OR CONTINUE
# ───────────────────────────────────────────────────────
if st.session_state.stage == "choose":
    st.markdown("### 🚀 What would you like to do?")
    st.markdown("Choose between creating a new RAG pipeline or continuing a previous conversation.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_new, col_continue = st.columns(2)
    
    with col_new:
        st.markdown("""
        <div style="background:#f0fdf4; border:2px solid #10b981; border-radius:12px; padding:2rem; text-align:center;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">📄</div>
            <div style="font-weight:700; font-size:1.2rem; margin-bottom:0.5rem; color:#166534;">Create New Pipeline</div>
            <div style="font-size:0.9rem; color:#059669; margin-bottom:1.5rem;">Upload documents and build a new RAG pipeline from scratch</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("➕ Create New", key="btn_new", use_container_width=True, type="primary"):
            st.session_state.pipeline_mode = "new"
            st.session_state.stage = "upload"
            st.rerun()
    
    with col_continue:
        available_pipelines = get_available_pipelines()
        
        if available_pipelines:
            st.markdown("""
            <div style="background:#fef3c7; border:2px solid #f59e0b; border-radius:12px; padding:2rem; text-align:center;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🔄</div>
                <div style="font-weight:700; font-size:1.2rem; margin-bottom:0.5rem; color:#92400e;">Continue Existing</div>
                <div style="font-size:0.9rem; color:#b45309; margin-bottom:1.5rem;">Pick up where you left off with a previous conversation</div>
            </div>
            """, unsafe_allow_html=True)
            
            selected_thread = st.selectbox(
                "Select a conversation:",
                available_pipelines,
                format_func=lambda x: f"📌 {x[:8]}... (Thread ID)",
                key="thread_selector"
            )
            
            if st.button("▶️ Continue", key="btn_continue", use_container_width=True, type="primary"):
                st.session_state.pipeline_mode = "continue"
                st.session_state.selected_thread_id = selected_thread
                st.session_state.stage = "load_continue"
                st.rerun()
        else:
            st.markdown("""
            <div style="background:#f3f4f6; border:2px solid #d1d5db; border-radius:12px; padding:2rem; text-align:center;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">📭</div>
                <div style="font-weight:700; font-size:1.2rem; margin-bottom:0.5rem; color:#4b5563;">No Previous Pipelines</div>
                <div style="font-size:0.9rem; color:#6b7280; margin-bottom:1.5rem;">Create a new pipeline to get started</div>
            </div>
            """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────
# STAGE 0.5: LOAD CONTINUE - Load from existing thread
# ───────────────────────────────────────────────────────
elif st.session_state.stage == "load_continue":
    st.markdown("### ⏳ Loading previous conversation...")
    
    # Allow user to override Qdrant collection name if needed
    st.info("📌 Specify the Qdrant collection used for this pipeline (default: 'documents')")
    col1, col2 = st.columns(2)
    with col1:
        qdrant_host_load = st.text_input("Qdrant Host", value=st.session_state.qdrant_host, key="load_host")
    with col2:
        qdrant_port_load = st.number_input("Qdrant Port", value=st.session_state.qdrant_port, step=1, key="load_port")
    
    # Fetch available collections
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=qdrant_host_load, port=int(qdrant_port_load))
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        if not collection_names:
            collection_names = [st.session_state.collection_name]
            st.warning("No collections found in Qdrant.")
    except Exception as e:
        st.warning(f"Could not connect to Qdrant: {e}")
        collection_names = [st.session_state.collection_name]
    
    qdrant_collection = st.selectbox("Collection Name", collection_names, index=0 if st.session_state.collection_name not in collection_names else collection_names.index(st.session_state.collection_name), key="load_collection")
    
    if st.button("✅ Load Pipeline", type="primary", use_container_width=True):
        with st.spinner("Loading pipeline from PostgreSQL..."):
            try:
                # We need embedding and reranker models to load the pipeline
                from sentence_transformers import SentenceTransformer, CrossEncoder
                
                with st.spinner("Loading models..."):
                    emb_model = SentenceTransformer("BAAI/bge-small-en")
                    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    
                    # Restore vector store with user-specified collection name
                    vector_store = QdrantVectorStore(
                        collection_name=qdrant_collection,
                        host=qdrant_host_load,
                        port=int(qdrant_port_load),
                        vector_size=384
                    )
                    
                    retriever = DenseQdrantRetriever(emb_model, reranker, vector_store)
                    
                    # Load the graph with the saved thread_id
                    graph, lg_state, thread_id, pool = load_pipeline_from_thread(
                        st.session_state.selected_thread_id,
                        retriever,
                        llm_backend if 'llm_backend' in locals() else "Groq",
                        groq_api_key if 'groq_api_key' in locals() else None,
                        groq_model if 'groq_model' in locals() else "llama-3.1-8b-instant",
                        ollama_model if 'ollama_model' in locals() else "mistral"
                    )
                    
                    if graph and lg_state is not None:
                        # Restore session state with the settings used
                        st.session_state.embedding_model = emb_model
                        st.session_state.reranker_model = reranker
                        st.session_state.vector_store = vector_store
                        st.session_state.retriever = retriever
                        st.session_state.lg_graph = graph
                        st.session_state.collection_name = qdrant_collection
                        st.session_state.qdrant_host = qdrant_host_load
                        st.session_state.qdrant_port = int(qdrant_port_load)
                        st.session_state.lg_state = lg_state
                        st.session_state.thread_id = thread_id
                        st.session_state.pg_pool = pool
                        
                        # Build chat history from lg_state messages (in pairs: user, assistant)
                        messages = lg_state.get("messages", [])
                        chat_history = []
                        for i, msg in enumerate(messages):
                            if i % 2 == 0:  # user message (even indices)
                                chat_history.append({"role": "user", "content": msg, "sources": []})
                            else:  # assistant message (odd indices)
                                chat_history.append({"role": "assistant", "content": msg, "sources": lg_state.get("sources", [])})
                        
                        st.session_state.chat_history = chat_history
                        st.session_state.stage = "chat"
                        st.success(f"✅ Loaded conversation with {len(messages)} messages!")
                        st.rerun()
                    else:
                        st.error("Failed to load pipeline.")
                        if st.button("⬅️ Go Back"):
                            st.session_state.stage = "choose"
                            st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading pipeline: {e}")
                st.exception(e)
                if st.button("⬅️ Go Back"):
                    st.session_state.stage = "choose"
                    st.rerun()


# ───────────────────────────────────────────────────────
# STAGE 1: UPLOAD
# ───────────────────────────────────────────────────────
if st.session_state.stage == "upload":
    st.markdown("### 📁 Upload Your Documents")
    st.info("Upload one or more PDF, DOCX, or JSON files from your folder.")

    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, JSON)",
        type=["pdf", "docx", "json"],
        accept_multiple_files=True,
        help="You can upload multiple files at once."
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = f.size / 1024
            st.markdown(f"- 📄 `{f.name}` — {size_kb:.1f} KB")

        if st.button("✅ Ingest Documents", type="primary", use_container_width=True):
            with st.spinner("📖 Reading documents..."):
                # Save to temp dir
                tmp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    dest = os.path.join(tmp_dir, uf.name)
                    with open(dest, "wb") as out:
                        out.write(uf.getbuffer())

                # Ingest
                ingestor = DocumentIngestor(tmp_dir)
                docs = ingestor.ingest()

                if not docs:
                    st.error("No documents could be read. Please check the file formats.")
                else:
                    # Preprocess
                    with st.spinner("✂️ Chunking and cleaning text..."):
                        import nltk
                        nltk.download("punkt", quiet=True)
                        nltk.download("punkt_tab", quiet=True)
                        processor = TextPreprocessor(chunk_size=500, overlap=50)
                        chunks = processor.process_documents(docs)

                    st.session_state.docs = docs
                    st.session_state.chunks = chunks
                    st.session_state.ingested_files = [d["file_name"] for d in docs]
                    st.session_state.stage = "vectorize"

                    st.success(f"✅ Ingested {len(docs)} document(s) → {len(chunks)} chunks created!")
                    st.rerun()


# ───────────────────────────────────────────────────────
# STAGE 2: VECTORIZE
# ───────────────────────────────────────────────────────
elif st.session_state.stage == "vectorize":
    st.markdown("### 🔢 Vectorize & Store in Qdrant")

    st.markdown(f"""
    <div class="success-box">
        ✅ <b>{len(st.session_state.docs)} documents</b> ingested → 
        <b>{len(st.session_state.chunks)} chunks</b> ready
    </div><br>
    """, unsafe_allow_html=True)

    st.markdown("**Files ready:**")
    for f in st.session_state.ingested_files:
        st.markdown(f"- 📄 `{f}`")

    st.markdown("---")
    st.markdown("Click below to embed all chunks and store them in Qdrant:")

    save_graph_image = st.checkbox("Save LangGraph pipeline image", value=False)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if st.button("🚀 Embed & Store in Qdrant", type="primary", use_container_width=True):
            try:
                from sentence_transformers import SentenceTransformer, CrossEncoder

                # Load models
                with st.spinner(f"⏳ Loading embedding model `{embedding_model_name}`..."):
                    emb_model = SentenceTransformer(embedding_model_name)
                    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

                # Embed chunks
                chunks = st.session_state.chunks
                progress_bar = st.progress(0, text="Embedding chunks...")

                for i, chunk in enumerate(chunks):
                    chunk["embedding"] = emb_model.encode(
                        chunk["content"],
                        normalize_embeddings=True
                    )
                    progress_bar.progress((i + 1) / len(chunks), text=f"Embedding chunk {i+1}/{len(chunks)}")

                progress_bar.empty()

                # Init Qdrant & insert
                with st.spinner("📦 Connecting to Qdrant and inserting vectors..."):
                    vsize = vector_size_map.get(embedding_model_name, 384)
                    vector_store = QdrantVectorStore(
                        collection_name=collection_name,
                        host=qdrant_host,
                        port=int(qdrant_port),
                        vector_size=vsize
                    )
                    vector_store.insert(chunks)

                # Build retriever
                retriever = DenseQdrantRetriever(emb_model, reranker, vector_store)

                # Build LangGraph graph
                graph = build_graph(
                    retriever=retriever,
                    llm_backend=llm_backend,
                    groq_api_key=groq_api_key if llm_backend == "Groq" else None,
                    groq_model=groq_model if llm_backend == "Groq" else "llama3-70b-8192",
                    ollama_model=ollama_model if llm_backend == "Ollama (Local)" else "mistral"
                )

                # Save graph image
                if save_graph_image:
                    try:
                        with open("langgraph_pipeline.png", "wb") as f:
                            f.write(graph.get_graph().draw_png())
                        st.info("✅ Graph image saved as 'langgraph_pipeline.png'")
                    except ImportError as e:
                        st.warning(f"⚠️ Could not save graph image: {e}. Install pygraphviz to enable graph visualization.")

                # Save to session
                st.session_state.embedding_model = emb_model
                st.session_state.reranker_model = reranker
                st.session_state.vector_store = vector_store
                st.session_state.retriever = retriever
                st.session_state.lg_graph = graph
                st.session_state.lg_state = {"messages": [], "summary": "", "sources": []}
                st.session_state.total_chunks = len(chunks)
                st.session_state.collection_name = collection_name
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_port = int(qdrant_port)
                st.session_state.stage = "chat"

                st.success(f"✅ {len(chunks)} chunks embedded and stored in Qdrant collection `{collection_name}`!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during vectorization: {e}")
                st.exception(e)

    with col_b:
        if st.button("⬅️ Re-upload Files", use_container_width=True):
            st.session_state.stage = "upload"
            st.session_state.docs = []
            st.session_state.chunks = []
            st.rerun()


# ───────────────────────────────────────────────────────
# STAGE 3: CHAT  (powered by LangGraph)
# ───────────────────────────────────────────────────────
elif st.session_state.stage == "chat":
    st.markdown("### 💬 Chat with Your Documents")

    # ── Sidebar: active graph info ───────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧠 LangGraph State")
        n_msgs    = len(st.session_state.lg_state.get("messages", []))
        has_summary = bool(st.session_state.lg_state.get("summary", "").strip())
        st.metric("Messages in memory", n_msgs)
        st.caption("✅ Summary active" if has_summary else "⏳ Summary triggers after 8 msgs")
        if has_summary:
            with st.expander("📝 Current Summary"):
                st.caption(st.session_state.lg_state["summary"])

    # ── Chat history display ─────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; color:#94a3b8; padding:2rem;">
                <div style="font-size:3rem">💬</div>
                <div>Ask anything about your uploaded documents!</div>
                <div style="font-size:0.8rem; margin-top:0.5rem; color:#cbd5e1;">
                    Powered by LangGraph · RAG + Dense Retrieval + CrossEncoder Reranking
                </div>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-end; margin-bottom:0.5rem;">
                    <div class="chat-bubble-user">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-start; margin-bottom:0.3rem;">
                    <div class="chat-bubble-bot">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)

                # Show RAG sources
                if msg.get("sources"):
                    with st.expander("📚 View Sources", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            fname = src.get("metadata", {}).get("file_name", "Unknown")
                            score = src.get("score", 0)
                            st.markdown(f"""
                            <span class="source-tag">#{i} {fname}</span>
                            <span style="color:#64748b; font-size:0.78rem;">score: {score:.3f}</span>
                            <div style="background:#f8fafc; border-left:3px solid #6366f1; padding:0.5rem 0.8rem;
                                        border-radius:4px; margin:0.3rem 0; font-size:0.85rem; color:#374151;">
                                {src['content'][:300]}{'...' if len(src['content']) > 300 else ''}
                            </div>
                            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Input area ───────────────────────────────────
    if st.session_state.get("clear_query_input", False):
        st.session_state.query_input = ""
        st.session_state.clear_query_input = False

    with st.form(key="chat_form"):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_query = st.text_input(
                "Ask a question",
                placeholder="What is the approval flow? / Who approves phase 2?",
                label_visibility="collapsed",
                key="query_input"
            )
        with col_btn:
            send = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

    if send and user_query.strip():
        # Validate LLM config
        if llm_backend == "Groq" and not groq_api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
        else:
            # Append user message to display history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })

            with st.spinner("🔍 Retrieving → Reranking → Generating..."):
                try:
                    # ── LangGraph invoke ────────────────────────
                    # 1. Push user message into LangGraph state
                    lg_state = st.session_state.lg_state
                    lg_state["messages"].append(user_query)

                    # 2. Run the graph: chat_node → summarize_node → END
                    graph = st.session_state.lg_graph
                    # lg_state = graph.invoke(lg_state) # original without postgres config
                    try:
                        lg_state = graph.invoke(
                            lg_state,
                            config={
                                "configurable": {
                                    "thread_id": st.session_state.thread_id
                                }
                            }
                        )
                    except Exception as invoke_err:
                        if "connection" in str(invoke_err).lower():
                            st.error("⚠️ PostgreSQL connection lost. Make sure PostgreSQL is running at localhost:5433")
                        raise
                    
                    # 3. Persist updated state
                    st.session_state.lg_state = lg_state

                    # 4. Extract answer (last message) and sources
                    answer  = lg_state["messages"][-1]
                    sources = lg_state.get("sources", [])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = str(e)
                    if "connection" in error_msg.lower():
                        error_msg = "PostgreSQL connection error. Ensure PostgreSQL is running at localhost:5433"
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error: {error_msg}",
                        "sources": []
                    })
                    st.error(f"Full error details: {e}")
                    import traceback
                    st.error(traceback.format_exc())

            # Clear the chat input after sending one query on next rerun
            st.session_state.clear_query_input = True
            st.rerun()

    # ── Footer controls ──────────────────────────────
    col_clr, col_rst = st.columns([1, 1])
    with col_clr:
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                # Reset LangGraph state too (fresh conversation)
                st.session_state.lg_state = {"messages": [], "summary": "", "sources": []}
                st.rerun()
    with col_rst:
        if st.button("🔄 New Pipeline", use_container_width=True):
            st.session_state.stage = "choose"
            st.rerun()
