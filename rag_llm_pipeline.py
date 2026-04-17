from typing import List, Dict, TypedDict

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
