# Semantic Search with Fuzzy Clustering and Semantic Cache
A lightweight semantic search system for the 20 Newsgroups dataset combining vector embeddings, probabilistic clustering, and a custom semantic cache exposed via FastAPI.

## Project Overview

This project builds a lightweight semantic search engine for the 20 Newsgroups dataset.

The system:

- Converts documents into semantic embeddings
- Groups documents using probabilistic (fuzzy) clustering
- Implements a semantic cache to reuse results for similar queries
- Exposes the system through a FastAPI service

The goal is to demonstrate how semantic similarity can reduce redundant computation in search systems. 

## System Architecture

User Query → Query Embedding (SentenceTransformer) → Semantic Cache  
        ↳ Cache Hit → Return Cached Result  
        ↳ Cache Miss → FAISS Vector Search → Retrieve Documents → Store Result in Cache

The system exposes this functionality through a FastAPI service.

## Part 1 — Embeddings and Vector Database

Embedding Model : 
I used the SentenceTransformer model all-MiniLM-L6-v2 as it provides:
- Strong Semantic Representations
- Fast Inference
- Compact Embeddings (384 dimensions)

Each document in the corpus is converted into an embedding vector.

Vector Database : 
The embeddings are indexed using FAISS,which enables efficient similarity search across thousands of documents.
FAISS allows fast nearest-neighbor retrieval for semantic queries.

## Part 2 — Fuzzy Clustering

Instead of assigning each document to a single cluster,the system uses Gaussian Mixture Models (GMM) to generate probabilistic cluster memberships.
Each document receives a probability distribution across clusters.

Example:
Cluster 1 → 0.12
Cluster 5 → 0.34
Cluster 8 → 0.27
This reflects the reality that many documents belong to multiple semantic topics.

Boundary Case Analysis : 
To identify ambiguous documents,I analyzed the maximum cluster probability per document.
Documents with the lowest max probability represent boundary cases between clusters.

Example boundary case:
Max cluster probability ≈ 0.17

This indicates that the document lies near the boundary between multiple semantic clusters rather than belonging clearly to a single topic.

## Part 3 - Semantic Cache

Traditional caches only match exact queries.
This project implements a semantic cache that detects similar queries even if they are phrased differently.

The cache stores:
- query text
- query embedding
- retrieved result
- cluster information

When a new query arrives:
- The query is embedded.
- Cosine similarity is computed against cached query embeddings.
- If similarity exceeds a threshold, the cached result is returned.

**Tunable Similarity Threshold :**
The cache includes a similarity threshold parameter.

Example behavior:
Threshold = 0.9 → very strict, almost no cache hits
Threshold = 0.7 → captures paraphrased queries

Example:
Query 1:
"space shuttle launch"

Query 2:
"NASA shuttle mission"

Similarity ≈ 0.74
With a threshold of 0.7, the second query correctly triggers a cache hit, avoiding redundant vector search.

PART 4 — FAST API SERVICE
 ------------------------
The system exposes three endpoints:

POST /query
Processes a user query, checks the semantic cache, and returns results.

GET /cache/stats
Returns cache statistics including hit rate.

DELETE /cache
Clears the cache and resets statistics.

The API can be tested through the automatically generated FastAPI Swagger interface.

**Key Design Decisions :**

SentenceTransformers were chosen for high-quality semantic embeddings.
FAISS was selected for efficient vector similarity search.
Gaussian Mixture Models were used to produce probabilistic cluster memberships rather than hard assignments.
A custom semantic cache was implemented from first principles without external caching libraries, as required.

## Repository Structure 

app/
  api/            FastAPI endpoints
  cache/          Semantic cache implementation
  clustering/     Fuzzy clustering logic
  embeddings/     Embedding model utilities

scripts/
  load_dataset.py        Dataset preparation
  analyze_clusters.py    Boundary case analysis

data/
  newsgroups.csv
  embeddings.npy
  cluster_membership.npy

main.py                  FastAPI entry point
requirements.txt         Project dependencies

**Running the Project :**

 - Create a virtual environment: python -m venv venv
 - Activate the environment and install dependencies: pip install -r requirements.txt
 - Run the API server: uvicorn main:app --reload
 - Open the API documentation: http://127.0.0.1:8000/docs

**Future Improvements :**

Cluster-aware cache indexing to reduce lookup complexity.
Adaptive cache thresholds based on query distribution.
More advanced clustering analysis using topic modeling.

**Conclusion :**  

This project demonstrates how semantic embeddings, fuzzy clustering, and a custom semantic cache can work together to build an efficient semantic search system.
The system avoids redundant computation by recognizing semantically similar queries and reusing previously computed results.