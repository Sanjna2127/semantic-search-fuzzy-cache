import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv("data/newsgroups.csv")

print("Loading cluster memberships...")
membership = np.load("data/cluster_membership.npy")

# maximum cluster probability per document
max_probs = membership.max(axis=1)

# get indices of most ambiguous documents
ambiguous_indices = max_probs.argsort()[:5]

print("\nMost ambiguous documents:\n")

for idx in ambiguous_indices:
    print("Max cluster probability:", max_probs[idx])
    print(df.iloc[idx]["text"][:400])
    print("\n------------------------\n") 