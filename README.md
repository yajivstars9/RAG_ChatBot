<<<<<<< HEAD
# RAG_ChatBot
=======
# 🧠 RAG Document Chatbot

A Streamlit app to upload documents, vectorize them with Qdrant, and chat using dense retrieval + reranking.


## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Qdrant (Docker required)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Run the app
```bash
streamlit run rag_app.py
```

---

## 🔄 App Flow

| Step | What happens |
|------|-------------|
| **① Upload** | Upload PDF, DOCX, or JSON files. They are ingested, cleaned, and chunked. |
| **② Vectorize** | Chunks are embedded with `BAAI/bge-small-en` (or your chosen model) and stored in Qdrant. |
| **③ Chat** | Ask questions. The app retrieves top chunks via cosine similarity, reranks with a CrossEncoder, and generates answers via Groq or Ollama. |

---

## ⚙️ Configuration (Sidebar)

| Setting | Default | Notes |
|---------|---------|-------|
| LLM Backend | Groq | Switch to Ollama for fully local usage |
| Groq API Key | — | Get one at console.groq.com |
| Ollama Model | mistral | Make sure Ollama is running locally |
| Embedding Model | BAAI/bge-small-en | bge-large-en gives better quality |
| Qdrant Host/Port | localhost:6333 | Point to your Qdrant instance |
| Collection Name | documents | One collection per session |

---

## 📦 Supported File Types
- **PDF** — via `pypdf`
- **DOCX** — via `python-docx`
- **JSON** — parsed and stringified

---

## 🧩 Architecture

```
Files → DocumentIngestor → TextPreprocessor (clean + chunk)
     → SentenceTransformer (embed) → QdrantVectorStore (upsert)
     → DenseQdrantRetriever (cosine search → CrossEncoder rerank)
     → build_prompt → Groq / Ollama → Answer
```


Flow : Each query → retrieve knowledge → generate answer → compress memory → repeat

   ┌──────────────┐
   │   Retriever  │  (knowledge)
   └──────┬───────┘
          ↓
User → Chat Node → LLM → Answer
         ↑
   ┌──────────────┐
   │   Summary    │  (long-term memory)
   └──────────────┘
         ↑
   ┌──────────────┐
   │  Messages    │  (short-term memory)
   └──────────────┘


checkpointer = PostgresSaver(pool)
checkpointer.setup() # imp to run for postgres saver to run first time.
>>>>>>> b7582b8 (Initial commit)
