"""
industry_prox.py: Industry proximity scoring (I), maps SIC/GICS to buckets and distances.
"""
import yaml
import os
import numpy as np

INDUSTRY_MAP_PATH = os.path.join(os.path.dirname(__file__), '../../config/industry_map.yaml')


def load_industry_map():
    """Load industry mapping."""
    if os.path.exists(INDUSTRY_MAP_PATH):
        with open(INDUSTRY_MAP_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}


def score_industry_proximity(target_industry, candidate_industry, industry_map=None):
    """
    Compute industry proximity score I.
    Returns score in [0, 1].
    """
    if not target_industry or not candidate_industry:
        return 0.5  # Neutral if missing
    
    if industry_map is None:
        industry_map = load_industry_map()
    
    # Simple matching: if industries are similar, return high score
    target_lower = target_industry.lower()
    candidate_lower = candidate_industry.lower()
    
    # Exact match
    if target_lower == candidate_lower:
        return 1.0
    
    # Partial match
    if target_lower in candidate_lower or candidate_lower in target_lower:
        return 0.8
    
    # Check bucket distances if available
    bucket_dist = industry_map.get('bucket_dist', {})
    if bucket_dist:
        # Try to find matching buckets
        for bucket, distances in bucket_dist.items():
            if bucket.lower() in target_lower:
                # Check candidate against this bucket's distances
                for other_bucket, distance in distances.items():
                    if other_bucket.lower() in candidate_lower:
                        # Distance of 0 = same bucket, 1.0 = far
                        score = 1.0 - min(distance, 1.0)
                        return float(np.clip(score, 0.0, 1.0))
    
    # Default: low similarity
    return 0.3


if __name__ == "__main__":
    # Test
    target = "Technology"
    candidate = "Software - Infrastructure"
    score = score_industry_proximity(target, candidate)
    print(f"Industry proximity: {score}")
