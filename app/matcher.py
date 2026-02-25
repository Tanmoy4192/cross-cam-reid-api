import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.6

def match_embeddings(emb1, emb2):
    matches = []
    unmatched = []

    if len(emb1) == 0 or len(emb2) == 0:
        return matches, unmatched

    sim_matrix = cosine_similarity(emb1, emb2)

    used_j = set()

    for i in range(len(emb1)):
        j = np.argmax(sim_matrix[i])
        if sim_matrix[i][j] > THRESHOLD and j not in used_j:
            matches.append((i, j, sim_matrix[i][j]))
            used_j.add(j)
        else:
            unmatched.append(("image1", i))

    for j in range(len(emb2)):
        if j not in used_j:
            unmatched.append(("image2", j))

    return matches, unmatched