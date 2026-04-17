import os
import json
import re
import tempfile
import shutil
import numpy as np
import streamlit as st
from typing import List, Dict, TypedDict
import logging
import warnings

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
# MODULE 1: DOCUMENT INGESTOR
# ═══════════════════════════════════════════════════════
class DocumentIngestor:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def ingest(self) -> List[Dict]:
        documents = []

        for root, _, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                try:
                    if file.endswith(".pdf"):
                        text = self._read_pdf(file_path)
                    elif file.endswith(".docx"):
                        text = self._read_docx(file_path)
                    elif file.endswith(".json"):
                        text = self._read_json(file_path)
                    else:
                        continue

                    documents.append({
                        "file_name": file,
                        "file_path": file_path,
                        "content": text
                    })

                except Exception as e:
                    st.warning(f"❌ Error processing {file}: {e}")

        return documents

    def _read_pdf(self, file_path: str) -> str:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

    def _read_docx(self, file_path: str) -> str:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _read_json(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════
# MODULE 2: TEXT PREPROCESSOR
# ═══════════════════════════════════════════════════════
class TextPreprocessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        processed_chunks = []

        for doc in documents:
            clean_text = self._clean_text(doc["content"])
            chunks = self._smart_chunk_with_overlap(clean_text)

            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    "chunk_id": f"{doc['file_name']}_{i}",
                    "file_name": doc["file_name"],
                    "file_path": doc["file_path"],
                    "content": chunk
                })

        return processed_chunks

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.encode("utf-8", "ignore").decode()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:%()-\[\]{}]", "", text)
        text = re.sub(r"([.,!?])\1+", r"\1", text)
        return text.strip()

    def _smart_chunk_with_overlap(self, text, max_size=500, overlap_sentences=1):
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        chunks = []
        current = []

        for sent in sentences:
            current.append(sent)
            joined = " ".join(current)

            if len(joined) > max_size:
                chunks.append(joined)
                current = current[-overlap_sentences:]

        if current:
            chunks.append(" ".join(current))

        return chunks


# ═══════════════════════════════════════════════════════
# MODULE 3: QDRANT VECTOR STORE
# ═══════════════════════════════════════════════════════
class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        vector_size: int = 384
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self._create_collection(vector_size)

    def _create_collection(self, vector_size):
        from qdrant_client.models import VectorParams, Distance

        if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def insert(self, chunks: List[Dict]):
        from qdrant_client.models import PointStruct

        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=i,
                    vector=chunk["embedding"].tolist(),
                    payload={
                        "content": chunk["content"],
                        "file_name": chunk.get("file_name", ""),
                        "chunk_id": chunk.get("chunk_id", str(i))
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector, top_k: int = 20, file_name: str = None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if file_name:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_name",
                        match=MatchValue(value=file_name)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        results = response.points
        return [
            {
                "content": r.payload.get("content", ""),
                "metadata": r.payload,
                "score": r.score
            }
            for r in results
        ]

    def collection_count(self):
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)


# ═══════════════════════════════════════════════════════
# MODULE 4: DENSE QDRANT RETRIEVER
# ═══════════════════════════════════════════════════════
class DenseQdrantRetriever:
    def __init__(self, model, reranker, vector_store):
        self.model = model
        self.reranker = reranker
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5):
        # Step 1: Embed query
        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Step 2: Vector DB search
        candidates = self.vector_store.search(query_vector, top_k=20)

        # Step 3: Rerank
        if not candidates:
            return []

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [item[0] for item in reranked[:top_k]]


# ═══════════════════════════════════════════════════════
# MODULE 5: LLM — GROQ & OLLAMA CALL
# ═══════════════════════════════════════════════════════

def call_llm(prompt: str, api_key: str, model: str = "llama3-70b-8192") -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


def call_llm_ollama(prompt: str, model: str = "mistral") -> str:
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.text}")
    data = response.json()
    if "response" not in data:
        raise Exception(f"Unexpected Ollama response: {data}")
    return data["response"]


# ═══════════════════════════════════════════════════════
# MODULE 6: LANGGRAPH — STATE + NODES + GRAPH
# ═══════════════════════════════════════════════════════

# ── 6a. State definition ─────────────────────────────
class ChatState(TypedDict):
    messages: List[str]    # flat list of interleaved user / assistant turns
    summary:  str          # rolling compressed conversation history
    sources:  List[Dict]   # RAG sources from the latest turn
    rewritten_query: str   # optional field to store rewritten query for RAG retrieval


# ── 6b. Rewrite node ────────────────────────────────
def make_rewrite_node(llm_backend: str,
                      groq_api_key: str = None, groq_model: str = "llama-3.1-8b-instant",
                      ollama_model: str = "mistral"):
    
    def rewrite_node(state: ChatState) -> ChatState:
        messages = list(state.get("messages", []))
        summary  = state.get("summary", "")

        user_input = messages[-1]

        prompt = f"""
        Rewrite the user query to be fully self-contained using conversation context.

        Conversation summary:
        {summary}

        Recent messages:
        {messages[-4:]}

        Original query:
        {user_input}

        Rewritten query:
        """

        if llm_backend == "Groq":
            rewritten = call_llm(prompt, api_key=groq_api_key, model=groq_model)
        else:
            rewritten = call_llm_ollama(prompt, model=ollama_model)

        # 👉 store rewritten query separately
        state["rewritten_query"] = rewritten.strip()
        return state

    return rewrite_node

# ── 6c. Chat node (RAG + LLM) ────────────────────────
def make_chat_node(retriever, llm_backend: str,
                   groq_api_key: str = None, groq_model: str = "llama3-70b-8192",
                   ollama_model: str = "mistral"):
    """
    Factory — captures retriever + LLM config in a closure so the node
    signature stays compatible with LangGraph (state → state).
    """
    def chat_node(state: ChatState) -> ChatState:
        messages = list(state.get("messages", []))
        summary  = state.get("summary", "")

        user_input = messages[-1]

        # Use rewritten query if available
        search_query = state.get("rewritten_query", user_input)

        # RAG: retrieve top-3 reranked chunks
        results = retriever.search(search_query)[:3]
        context = "\n\n".join([r["content"] for r in results])

        prompt = f"""
            You are an assistant.

            Conversation summary:
            {summary}

            Recent messages:
            {messages[-4:]}

            Context:
            {context}

            Answer clearly in 2-3 sentences.

            User: {user_input}
            """

        if llm_backend == "Groq":
            answer = call_llm(prompt, api_key=groq_api_key, model=groq_model)
        else:
            answer = call_llm_ollama(prompt, model=ollama_model)

        messages.append(answer)

        return {
            "messages": messages,
            "summary":  summary,
            "sources":  results
        }

    return chat_node


# ── 6d. Summarize node ─────────────────────────────── > Summarized State(LG)
def make_summarize_node(llm_backend: str,
                        groq_api_key: str = None, groq_model: str = "llama3-70b-8192",
                        ollama_model: str = "mistral"):
    def summarize_node(state: ChatState) -> ChatState:
        messages = list(state["messages"])
        summary  = state.get("summary", "")
        sources  = state.get("sources", [])

        if len(messages) > 8:
            history = "\n".join(messages[:-4])

            prompt = f"""
                Summarize the conversation briefly.

                Previous summary:
                {summary}

                Conversation:
                {history}

                New summary:
                """
            if llm_backend == "Groq":
                summary = call_llm(prompt, api_key=groq_api_key, model=groq_model)
            else:
                summary = call_llm_ollama(prompt, model=ollama_model)

            messages = messages[-4:]

        return {
            "messages": messages,
            "summary":  summary,
            "sources":  sources
        }

    return summarize_node


# ── 6e. Graph builder ────────────────────────────────

def build_graph(retriever, llm_backend: str,
                groq_api_key: str = None, groq_model: str = "llama-3.1-8b-instant",
                ollama_model: str = "mistral"):
    
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo="postgresql://rag:rag@localhost:5433/RAG_storage",
        max_size=10
    )

    checkpointer = PostgresSaver(pool)
    
    # ── Nodes ───────────────────────────────
    rewrite_node   = make_rewrite_node(llm_backend, groq_api_key, groq_model, ollama_model)
    chat_node      = make_chat_node(retriever, llm_backend, groq_api_key, groq_model, ollama_model)
    summarize_node = make_summarize_node(llm_backend, groq_api_key, groq_model, ollama_model)

    builder = StateGraph(ChatState)

    builder.add_node("rewrite",   rewrite_node)
    builder.add_node("chat",      chat_node)
    builder.add_node("summarize", summarize_node)

    # ✅ NEW FLOW
    builder.set_entry_point("rewrite")
    builder.add_edge("rewrite",   "chat")
    builder.add_edge("chat",      "summarize")
    builder.add_edge("summarize", END)

    # ── Attach persistent memory ──────────────────
    return builder.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR CONTINUING FROM EXISTING PIPELINES
# ═══════════════════════════════════════════════════════

def get_available_pipelines():
    """Retrieve list of available chat threads from PostgreSQL."""
    try:
        from psycopg_pool import ConnectionPool
        import json
        
        pool = ConnectionPool(
            conninfo="postgresql://rag:rag@localhost:5433/RAG_storage",
            max_size=5
        )
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Query distinct thread_ids and get latest checkpoint info
                cur.execute("""
                    SELECT DISTINCT thread_id 
                    FROM checkpoints 
                    ORDER BY thread_id DESC 
                    LIMIT 50
                """)
                threads = cur.fetchall()
                
        pool.close()
        return [t[0] for t in threads] if threads else []
    except Exception as e:
        st.warning(f"Could not retrieve pipelines: {e}")
        return []


def load_pipeline_from_thread(thread_id: str, retriever, llm_backend: str,
                              groq_api_key: str = None, groq_model: str = "llama-3.1-8b-instant",
                              ollama_model: str = "mistral"):
    """Load an existing pipeline from PostgreSQL using thread_id."""
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool
        
        # Test connection first
        try:
            pool = ConnectionPool(
                conninfo="postgresql://rag:rag@localhost:5433/RAG_storage",
                max_size=10,
                timeout=5
            )
            # Test connection
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception as conn_err:
            raise Exception(f"PostgreSQL connection failed: {conn_err}. Make sure PostgreSQL is running at localhost:5433")
        
        checkpointer = PostgresSaver(pool)
        
        # Build the graph (same as new)
        rewrite_node   = make_rewrite_node(llm_backend, groq_api_key, groq_model, ollama_model)
        chat_node      = make_chat_node(retriever, llm_backend, groq_api_key, groq_model, ollama_model)
        summarize_node = make_summarize_node(llm_backend, groq_api_key, groq_model, ollama_model)

        builder = StateGraph(ChatState)
        builder.add_node("rewrite",   rewrite_node)
        builder.add_node("chat",      chat_node)
        builder.add_node("summarize", summarize_node)

        builder.set_entry_point("rewrite")
        builder.add_edge("rewrite",   "chat")
        builder.add_edge("chat",      "summarize")
        builder.add_edge("summarize", END)

        graph = builder.compile(checkpointer=checkpointer)
        
        # Load the previous state from checkpoint
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state_result = graph.get_state(config)
            lg_state = state_result.values if state_result else {"messages": [], "summary": "", "sources": []}
        except Exception as state_err:
            st.warning(f"Could not load previous state: {state_err}. Starting fresh.")
            lg_state = {"messages": [], "summary": "", "sources": []}
        
        return graph, lg_state, thread_id, pool
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None, None, None


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
        qdrant_collection = st.text_input("Collection Name", value=st.session_state.collection_name, key="load_collection")
    with col2:
        qdrant_host_load = st.text_input("Qdrant Host", value=st.session_state.qdrant_host, key="load_host")
    
    qdrant_port_load = st.number_input("Qdrant Port", value=st.session_state.qdrant_port, step=1, key="load_port")
    
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
