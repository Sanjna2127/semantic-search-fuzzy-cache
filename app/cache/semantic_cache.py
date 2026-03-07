import numpy as np


class SemanticCache:

    def __init__(self, similarity_threshold=0.9):
        self.cache = []
        self.similarity_threshold = similarity_threshold

        self.hit_count = 0
        self.miss_count = 0

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def lookup(self, query_vector):
        best_score = -1
        best_entry = None

        for entry in self.cache:
            score = self.cosine_similarity(query_vector, entry["vector"])
            print("Similarity with cached query:", score)

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.similarity_threshold:
            self.hit_count += 1
            return True, best_entry, best_score

        self.miss_count += 1
        return False, None, best_score

    def add(self, query, vector, result, cluster):
        self.cache.append({
            "query": query,
            "vector": vector,
            "result": result,
            "cluster": cluster
        })

    def stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0 