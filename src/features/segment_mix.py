"""
segment_mix.py: Segment mix similarity (M), cosine or fallback.
"""
import numpy as np
from collections import Counter


def score_segment_mix(target_mix, candidate_mix):
    """
    Compute segment mix similarity M using cosine similarity.
    Returns score in [0, 1].
    """
    if not target_mix or not candidate_mix:
        return 0.5  # Neutral if missing
    
    # Convert to vectors
    all_segments = set(target_mix.keys()) | set(candidate_mix.keys())
    
    target_vec = np.array([target_mix.get(s, 0) for s in all_segments])
    candidate_vec = np.array([candidate_mix.get(s, 0) for s in all_segments])
    
    # Cosine similarity
    dot_product = np.dot(target_vec, candidate_vec)
    norm_target = np.linalg.norm(target_vec)
    norm_candidate = np.linalg.norm(candidate_vec)
    
    if norm_target == 0 or norm_candidate == 0:
        return 0.5
    
    cosine = dot_product / (norm_target * norm_candidate)
    
    # Map from [-1, 1] to [0, 1]
    score = (cosine + 1) / 2.0
    
    return float(score)


if __name__ == "__main__":
    # Test
    target = {"Healthcare": 0.5, "Education": 0.3, "Commercial": 0.2}
    candidate = {"Healthcare": 0.6, "Education": 0.2, "Commercial": 0.2}
    score = score_segment_mix(target, candidate)
    print(f"Segment mix score: {score}")
