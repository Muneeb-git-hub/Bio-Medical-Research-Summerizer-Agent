import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def novelty_score(sentences):
    embeddings = model.encode(sentences)
    scores = np.zeros(len(sentences))

    for i in range(len(sentences)):
        # Compute cosine similarities of the current sentence with all others
        cosine_similarities = np.dot(embeddings, embeddings[i]) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings[i]))
        # Compute novelty score as 1 - average similarity
        scores[i] = 1 - np.mean(cosine_similarities)
    
    # Normalize scores between 0 and 1
    return np.clip(scores, 0, 1)
