"""
retrieval.py
The "R" in RAG. Embeds every dish description into a vector space and finds the
most similar dishes to a query using cosine similarity.

NOTE ON IMPLEMENTATION:
This uses a brute-force NumPy similarity search rather than a vector-database
library like FAISS. For a dataset this size (hundreds to low thousands of items),
brute force is actually the right engineering choice -- it's a single matrix
multiply, sub-millisecond, has zero native-library dependencies (so it can't
segfault from binary incompatibilities the way faiss/lightgbm sometimes do on
certain platforms), and is exactly what FAISS's `IndexFlatIP` (its simplest,
exact index type) does internally anyway. In an interview, be ready to explain
when this DOES stop being sufficient: once you have hundreds of thousands to
millions of vectors, brute force becomes too slow per query, and you'd move to
an approximate nearest-neighbor index (FAISS's IVF/HNSW, ScaNN, or a managed
vector DB like Pinecone/Weaviate) that trades a small amount of recall for
much faster lookup at scale.

NOTE ON EMBEDDINGS:
This uses TF-IDF + TruncatedSVD (a form of Latent Semantic Analysis) to produce
dense vectors with zero heavy dependencies. On your own laptop, you can swap in
real sentence-transformer embeddings for stronger semantic matching -- see the
commented class below. The similarity search code doesn't care which embedding
produced the vectors, which is worth mentioning: retrieval and embedding are
cleanly decoupled layers.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class DishRetriever:
    def __init__(self, dishes_df: pd.DataFrame, embedding_dim: int = 128):
        self.dishes_df = dishes_df.reset_index(drop=True)
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.corpus_embeddings = None  # shape: (n_dishes, embedding_dim), L2-normalized
        self._build_index()

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return matrix / norms

    def _embed_corpus(self, texts):
        tfidf = self.vectorizer.fit_transform(texts)
        embeddings = self.svd.fit_transform(tfidf)
        return self._normalize(embeddings).astype("float32")

    def _embed_query(self, text):
        tfidf = self.vectorizer.transform([text])
        embedding = self.svd.transform(tfidf)
        return self._normalize(embedding).astype("float32")

    def _build_index(self):
        texts = self.dishes_df["description"].tolist()
        self.corpus_embeddings = self._embed_corpus(texts)

    def retrieve(self, query: str, top_k: int = 20) -> pd.DataFrame:
        query_vec = self._embed_query(query)  # shape: (1, embedding_dim)
        # Since both sides are L2-normalized, dot product == cosine similarity.
        # This single matrix multiply is the entire "search" step.
        scores = self.corpus_embeddings @ query_vec[0]  # shape: (n_dishes,)
        top_indices = np.argsort(-scores)[:top_k]
        results = self.dishes_df.iloc[top_indices].copy()
        results["retrieval_score"] = scores[top_indices]
        return results.reset_index(drop=True)


# --- Optional drop-in replacement (requires: pip install sentence-transformers) ---
# Same brute-force cosine similarity pattern, just swapping the embedding source.
# from sentence_transformers import SentenceTransformer
#
# class DishRetrieverTransformer:
#     def __init__(self, dishes_df, model_name="all-MiniLM-L6-v2"):
#         self.dishes_df = dishes_df.reset_index(drop=True)
#         self.model = SentenceTransformer(model_name)
#         self.corpus_embeddings = self.model.encode(
#             self.dishes_df["description"].tolist(), normalize_embeddings=True
#         )
#
#     def retrieve(self, query, top_k=20):
#         q_emb = self.model.encode([query], normalize_embeddings=True)[0]
#         scores = self.corpus_embeddings @ q_emb
#         top_indices = np.argsort(-scores)[:top_k]
#         results = self.dishes_df.iloc[top_indices].copy()
#         results["retrieval_score"] = scores[top_indices]
#         return results.reset_index(drop=True)
#
# For production scale (100K+ items), replace the brute-force @ with a FAISS
# IVF/HNSW index or a managed vector DB -- the interface above stays the same.


if __name__ == "__main__":
    dishes = pd.read_csv("dishes.csv")
    retriever = DishRetriever(dishes)
    results = retriever.retrieve("something spicy and cheap under 200", top_k=5)
    print(results[["dish_name", "restaurant_name", "price", "spice_level", "retrieval_score"]])
