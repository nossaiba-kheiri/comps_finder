"""
evidence_quality.py: Evidence quality scoring (E), weight by evidence type.
"""
import numpy as np


def score_evidence_quality(sources):
    """
    Compute evidence quality score E.
    Sources: list of dicts with 'type' field (10K, IR, site, wiki, etc.)
    Returns score in [0, 1].
    """
    if not sources:
        return 0.0
    
    # Quality weights
    weights = {
        '10K': 1.0,
        '20-F': 1.0,
        'IR': 0.8,
        'site': 0.6,
        'wiki': 0.4
    }
    
    # Combine as 1 - Π(1 - w_i)
    product = 1.0
    for source in sources:
        source_type = source.get('type', 'site').upper()
        if '10-K' in source_type or '10K' in source_type:
            w = weights.get('10K', 0.6)
        elif '20-F' in source_type:
            w = weights.get('20-F', 0.6)
        elif 'IR' in source_type:
            w = weights.get('IR', 0.6)
        elif 'WIKI' in source_type:
            w = weights.get('wiki', 0.6)
        else:
            w = weights.get('site', 0.6)
        
        product *= (1 - w)
    
    score = 1 - product
    return float(np.clip(score, 0.0, 1.0))


if __name__ == "__main__":
    # Test
    import numpy as np
    sources = [
        {'type': '10K', 'url': '...'},
        {'type': 'site', 'url': '...'}
    ]
    score = score_evidence_quality(sources)
    print(f"Evidence quality: {score}")
