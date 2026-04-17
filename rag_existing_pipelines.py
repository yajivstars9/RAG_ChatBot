from rag_llm_pipeline import *
from rag_ingestion_embedding import *

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
