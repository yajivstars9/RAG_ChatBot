import os
import json
import re
import streamlit as st
from typing import List, Dict

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
