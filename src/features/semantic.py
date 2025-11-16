"""
semantic.py: Semantic similarity (S), cosine between normalized embeddings.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../nlp'))
try:
    from embeddings import embed_text, get_cached_embedding
except ImportError:
    # Fallback
    def get_cached_embedding(text, run_with_openai=False):
        np.random.seed(hash(text) % (2**31))
        v = np.random.randn(3072).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


def score_semantic_similarity(target_embedding, candidate_embedding):
    """
    Compute semantic similarity S using cosine similarity.
    Both embeddings should be L2-normalized.
    Returns score in [0, 1].
    """
    if target_embedding is None or candidate_embedding is None:
        return 0.0
    
    # Ensure they're numpy arrays
    if not isinstance(target_embedding, np.ndarray):
        target_embedding = np.array(target_embedding)
    if not isinstance(candidate_embedding, np.ndarray):
        candidate_embedding = np.array(candidate_embedding)
    
    # Cosine similarity (dot product since normalized)
    cosine = np.dot(target_embedding, candidate_embedding)
    
    # Map from [-1, 1] to [0, 1]
    score = (cosine + 1) / 2.0
    
    return float(np.clip(score, 0.0, 1.0))


if __name__ == "__main__":
    # Test
    target_emb = get_cached_embedding("payment processing for banks")
    candidate_emb = get_cached_embedding("financial transaction services")
    score = score_semantic_similarity(target_emb, candidate_emb)
    print(f"Semantic similarity: {score}")
