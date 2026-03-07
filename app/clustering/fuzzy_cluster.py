import numpy as np
from sklearn.mixture import GaussianMixture


class FuzzyClusterer:

    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.model = GaussianMixture(
    n_components=20,
    covariance_type="diag",
    reg_covar=1e-2,
    random_state=42
)
        

    def fit(self, embeddings):
        self.model.fit(embeddings)

    def get_membership(self, embeddings):
        probabilities = self.model.predict_proba(embeddings)

        # smooth probabilities slightly to avoid hard assignments
        probabilities = probabilities + 1e-6
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        return probabilities