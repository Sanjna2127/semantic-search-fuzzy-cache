from pydantic import BaseModel
from fastapi import APIRouter
import numpy as np
import pandas as pd

from app.embeddings.embedder import Embedder, build_faiss_index
from app.cache.semantic_cache import SemanticCache
cluster_membership = np.load("data/cluster_membership.npy") 

router = APIRouter()

df = pd.read_csv("data/newsgroups.csv")
embeddings = np.load("data/embeddings.npy")

index = build_faiss_index(embeddings)

embedder = Embedder()
cache = SemanticCache(similarity_threshold=0.7)

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
def query_api(data: QueryRequest):

    query = data.query

    query_vector = embedder.encode([query])[0]

    hit, entry, score = cache.lookup(query_vector)

    if hit:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(score),
            "result": entry["result"]
        }

    distances, indices = index.search(
        query_vector.reshape(1, -1).astype("float32"), 5
    )

    results = [df.iloc[i]["text"][:200] for i in indices[0]]
    dominant_cluster = int(np.argmax(cluster_membership[indices[0][0]])) 

    cache.add(query, query_vector, results, cluster=None)

    return {
    "query": query,
    "cache_hit": hit,
    "matched_query": entry["query"] if hit else None,
    "similarity_score": score if hit else None,
    "dominant_cluster": dominant_cluster,
    "result": results
} 


@router.get("/cache/stats")
def cache_stats():
    return cache.stats()


@router.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}  