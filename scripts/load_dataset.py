import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 


import pandas as pd
from app.embeddings.embedder import Embedder

print("Loading dataset...")

df = pd.read_csv("data/newsgroups.csv")

texts = df["text"].tolist()

print("Number of documents:", len(texts))

embedder = Embedder()

print("Generating embeddings...")

embeddings = embedder.encode(texts)

print("Embeddings generated!")

import numpy as np

np.save("data/embeddings.npy", embeddings)

print("Embeddings saved to data/embeddings.npy") 