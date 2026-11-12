"""
embeddings.py - Step 2 of the RAG pipeline: text chunks -> TF-IDF vectors.
The matrix is stored SPARSE (scipy .npz) so the vector database is tiny.
TfidfVectorizer already L2-normalises rows, so cosine similarity = dot product.
"""
import os, pickle
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store")
VECTORIZER_PATH = os.path.join(STORE, "tfidf_vectorizer.pkl")
MATRIX_PATH     = os.path.join(STORE, "tfidf_matrix.npz")
CHUNKS_PATH     = os.path.join(STORE, "chunks.pkl")


def build_tfidf_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                 stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts).tocsr()   # rows are L2-normalised
    print(f"  Matrix shape: {matrix.shape}  (rows=chunks, cols=vocabulary)")
    return vectorizer, matrix


def save_embeddings(vectorizer, matrix, chunks):
    os.makedirs(STORE, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f: pickle.dump(vectorizer, f)
    sp.save_npz(MATRIX_PATH, matrix)
    with open(CHUNKS_PATH, "wb") as f: pickle.dump(chunks, f)


def load_embeddings():
    with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
    matrix = sp.load_npz(MATRIX_PATH).tocsr()
    with open(CHUNKS_PATH, "rb") as f: chunks = pickle.load(f)
    return vectorizer, matrix, chunks


def embed_query(query_text, vectorizer):
    return vectorizer.transform([query_text])          # sparse, L2-normalised


def is_vector_store_ready():
    return all(os.path.exists(p) for p in [VECTORIZER_PATH, MATRIX_PATH, CHUNKS_PATH])
