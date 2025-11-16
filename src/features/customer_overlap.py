"""
customer_overlap.py: Customer overlap scoring (C), using direct substring matching.
"""
def score_customer_overlap(target_customers, candidate_customer_segment, taxonomy=None):
    """
    Compute customer overlap score C.
    Returns score in [0, 1] and hit count.
    
    Args:
        target_customers: List of target customer strings
        candidate_customer_segment: List of candidate customer segment strings
        taxonomy: (Deprecated - not used) Taxonomy dict for customer matching
    """
    if not target_customers or not candidate_customer_segment:
        return 0.0, 0
    
    target_lower = [c.lower() for c in target_customers]
    candidate_lower = [cs.lower() for cs in candidate_customer_segment] if isinstance(candidate_customer_segment, list) else [str(candidate_customer_segment).lower()]
    
    hits = 0
    hit_customers = []
    
    # Direct matches
    for tc in target_lower:
        for cc in candidate_lower:
            if tc in cc or cc in tc:
                hits += 1
                hit_customers.append(tc)
                break
    
    # Note: Taxonomy-based matching removed - using direct substring matching only
    
    # Normalize: hits / max possible matches
    # Max possible = min(target customers, candidate segments)
    max_possible = min(len(target_customers), len(candidate_customer_segment))
    score = min(hits / max_possible, 1.0) if max_possible > 0 else 0.0
    
    return score, hits


if __name__ == "__main__":
    # Test
    target = ["Banks", "Retailers"]
    candidate = ["banks", "credit unions", "retail stores"]
    score, hits = score_customer_overlap(target, candidate)
    print(f"Score: {score}, Hits: {hits}")
