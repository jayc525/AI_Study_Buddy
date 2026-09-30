from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from study_buddy.text_utils import split_paragraphs


@dataclass
class RetrievedChunk:
    text: str
    score: float


class RAGPipeline:
    """PDF chunking and retrieval pipeline using semantic embeddings with FAISS.

    Falls back to TF-IDF if sentence-transformers or FAISS is not available.
    """

    def __init__(self, document_text: str, chunk_words: int = 120):
        self.chunks = split_paragraphs(document_text, max_words=chunk_words)
        self.use_embeddings = False
        self.embedder = None
        self.faiss_index = None

        # TF-IDF fallback
        self.vectorizer = None
        self.tfidf_matrix = None

        if self.chunks:
            self._build_index()

    def _build_index(self):
        """Try to build a FAISS semantic index; fall back to TF-IDF."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            chunk_embeddings = self.embedder.encode(
                self.chunks, show_progress_bar=False, convert_to_numpy=True
            )
            chunk_embeddings = chunk_embeddings.astype("float32")

            # Normalize for cosine similarity
            faiss.normalize_L2(chunk_embeddings)

            dimension = chunk_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(chunk_embeddings)
            self.use_embeddings = True

        except Exception:
            # Fall back to TF-IDF
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            self.use_embeddings = False

    def retrieve(self, question: str, top_k: int = 4) -> List[RetrievedChunk]:
        if not self.chunks:
            return []

        if self.use_embeddings and self.faiss_index is not None:
            return self._retrieve_semantic(question, top_k)
        return self._retrieve_tfidf(question, top_k)

    def _retrieve_semantic(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve chunks using FAISS semantic search."""
        import faiss

        query_embedding = self.embedder.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.chunks)))

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0 and score > 0:
                results.append(RetrievedChunk(text=self.chunks[index], score=float(score)))
        return results

    def _retrieve_tfidf(self, question: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve chunks using TF-IDF cosine similarity (fallback)."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        best_indexes = scores.argsort()[::-1][:top_k]

        return [
            RetrievedChunk(text=self.chunks[index], score=float(scores[index]))
            for index in best_indexes
            if scores[index] > 0
        ]

    def build_prompt(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"Source {index}: {chunk.text[:900]}"
            for index, chunk in enumerate(retrieved_chunks, start=1)
        )

        return f"""
Answer using only the uploaded PDF context.
Keep it short and student-friendly.
If the context does not contain the answer, say the notes do not contain enough information.
Do not show internal reasoning.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    def stats(self) -> str:
        method = "Semantic embeddings (FAISS + MiniLM)" if self.use_embeddings else "TF-IDF with cosine similarity"
        return f"RAG index ready. Chunks: {len(self.chunks)}. Retrieval: {method}."
