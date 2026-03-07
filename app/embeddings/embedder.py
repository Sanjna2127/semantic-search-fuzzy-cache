from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        # load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.model.encode(texts)
import faiss
import numpy as np


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    return index 