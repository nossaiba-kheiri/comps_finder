"""
segment_distribution.py: Compute customer segment distribution similarity using continuous vectors.

This module treats customer segments as weighted allocation (portfolio exposure), not binary tags.
Uses cosine similarity + entropy-based concentration penalty.

Generic - works for any industry (no hardcoding).
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple


def extract_segment_distribution(company_data: Dict) -> Dict[str, float]:
    """
    Extract segment distribution from company data.
    
    Handles multiple formats:
    1. segment_mix dict: {"Healthcare": 0.4, "Education": 0.3, "Commercial": 0.3}
    2. segments list with shares: [{"name": "Healthcare", "share": 0.4}, ...]
    3. customer_segment list (equal weights): ["Healthcare", "Education", "Commercial"]
    
    Args:
        company_data: Dict with segment_mix, segments, or customer_segment
    
    Returns:
        Dict mapping segment names (lowercase) to shares (0.0-1.0)
    """
    # Try segment_mix first (explicit distribution)
    segment_mix = company_data.get('segment_mix')
    # Handle None explicitly (LLM might return null/None)
    if segment_mix is not None and isinstance(segment_mix, dict) and len(segment_mix) > 0:
        # Normalize to lowercase keys and ensure values sum to 1.0
        result = {str(k).lower().strip(): float(v or 0.0) for k, v in segment_mix.items() if v}
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
            return result
    
    # Try product_mix as fallback (some target.json files use product_mix for segment distribution)
    # This is common when product_mix represents segment revenue breakdown
    product_mix = company_data.get('product_mix')
    if product_mix is not None and isinstance(product_mix, dict) and len(product_mix) > 0:
        # Check if product_mix looks like segment distribution (has segment-like keys)
        # Common patterns: "healthcare_consulting", "education_consulting", "commercial_consulting"
        # or "Healthcare", "Education", "Commercial"
        # Extract segment name from key (remove "_consulting", "_services", etc.)
        result = {}
        for key, value in product_mix.items():
            try:
                # Try to convert value to float (handle various formats)
                if value is None:
                    continue
                val_float = float(value) if not isinstance(value, (bool, str)) else float(value) if str(value).replace('.', '').replace('-', '').isdigit() else 0.0
                if val_float > 0:
                    # Normalize key: remove common suffixes and convert to lowercase
                    seg_name = str(key).lower().strip()
                    # Remove common suffixes
                    for suffix in ['_consulting', '_services', '_segment', '_business', '_revenue', '_segment_revenue']:
                        if seg_name.endswith(suffix):
                            seg_name = seg_name[:-len(suffix)]
                            break
                    result[seg_name] = val_float
            except (ValueError, TypeError):
                # Skip invalid values
                continue
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
            return result
    
    # Try segments list with shares
    segments = company_data.get('segments', [])
    if segments and isinstance(segments, list) and len(segments) > 0:
        # Check if first element is a dict with "name" and "share"
        if isinstance(segments[0], dict) and 'name' in segments[0]:
            result = {}
            for seg in segments:
                name = str(seg.get('name', '')).lower().strip()
                share = float(seg.get('share', 0.0) or 0.0)
                if name and share > 0:
                    result[name] = share
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            return result
    
    # Fallback: customer_segment list (equal weights)
    customer_segment = company_data.get('customer_segment', [])
    if customer_segment:
        if isinstance(customer_segment, str):
            customer_segment = [customer_segment]
        elif not isinstance(customer_segment, list):
            customer_segment = []
        
        # Equal weights
        segments_list = [str(s).lower().strip() for s in customer_segment if s]
        if segments_list:
            weight = 1.0 / len(segments_list)
            return {seg: weight for seg in segments_list}
    
    # Additional fallback: customer_industries (verticals served)
    # This is important for companies where customer_segment is empty but customer_industries has data
    customer_industries = company_data.get('customer_industries', [])
    if customer_industries:
        if isinstance(customer_industries, str):
            customer_industries = [customer_industries]
        elif not isinstance(customer_industries, list):
            customer_industries = []
        
        # Equal weights
        industries_list = [str(i).lower().strip() for i in customer_industries if i]
        if industries_list:
            weight = 1.0 / len(industries_list)
            return {ind: weight for ind in industries_list}
    
    # Additional fallback: primary_customer_types (from LLM extraction)
    # This is a more specific field that might have data even if customer_segment is empty
    primary_customer_types = company_data.get('primary_customer_types', [])
    if primary_customer_types:
        if isinstance(primary_customer_types, str):
            primary_customer_types = [primary_customer_types]
        elif not isinstance(primary_customer_types, list):
            primary_customer_types = []
        
        # Equal weights
        types_list = [str(t).lower().strip() for t in primary_customer_types if t]
        if types_list:
            weight = 1.0 / len(types_list)
            return {t: weight for t in types_list}
    
    # Last resort fallback: Try to infer from business description or summary
    # Look for common segment keywords in the company's description
    business_text = ''
    if 'business_description' in company_data:
        business_text = str(company_data.get('business_description', '')).lower()
    elif 'summary' in company_data:
        business_text = str(company_data.get('summary', '')).lower()
    elif 'raw_profile_text' in company_data:
        business_text = str(company_data.get('raw_profile_text', '')).lower()
    
    if business_text:
        # Common segment keywords to look for
        segment_keywords = {
            'healthcare': ['healthcare', 'health care', 'hospital', 'medical', 'health', 'pharmaceutical', 'biotech'],
            'education': ['education', 'university', 'college', 'school', 'academic', 'higher education'],
            'financial': ['financial', 'finance', 'banking', 'bank', 'investment', 'capital markets', 'wealth management'],
            'government': ['government', 'public sector', 'federal', 'state', 'municipal', 'defense', 'military'],
            'retail': ['retail', 'consumer', 'e-commerce', 'shopping'],
            'technology': ['technology', 'tech', 'software', 'it', 'information technology'],
            'manufacturing': ['manufacturing', 'industrial', 'production', 'factory'],
            'energy': ['energy', 'oil', 'gas', 'utilities', 'power'],
            'commercial': ['commercial', 'business', 'enterprise', 'corporate']
        }
        
        found_segments = []
        for segment, keywords in segment_keywords.items():
            if any(keyword in business_text for keyword in keywords):
                found_segments.append(segment)
        
        if found_segments:
            weight = 1.0 / len(found_segments)
            return {seg: weight for seg in found_segments}
    
    return {}


def build_segment_vocabulary(all_companies: List[Dict]) -> List[str]:
    """
    Build vocabulary of all unique segment labels across all companies.
    
    Generic - works for any industry, any number of segments.
    
    Args:
        all_companies: List of company dicts (target + all candidates)
    
    Returns:
        Sorted list of unique segment names (lowercase)
    """
    all_segments: Set[str] = set()
    
    for company in all_companies:
        dist = extract_segment_distribution(company)
        all_segments.update(dist.keys())
    
    return sorted(all_segments)


def segment_vector(segment_dist: Dict[str, float], vocabulary: List[str]) -> np.ndarray:
    """
    Convert segment distribution to vector using vocabulary.
    
    Args:
        segment_dist: Dict mapping segment names (lowercase) to shares
        vocabulary: List of all unique segment names (lowercase)
    
    Returns:
        numpy array of length len(vocabulary), normalized to sum to 1.0
    """
    vec = np.zeros(len(vocabulary), dtype=float)
    
    for i, seg_name in enumerate(vocabulary):
        vec[i] = float(segment_dist.get(seg_name, 0.0) or 0.0)
    
    # Normalize to sum to 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity in [0, 1]
    """
    if len(a) != len(b):
        return 0.0
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    dot_product = np.dot(a, b)
    return float(dot_product / (norm_a * norm_b))


def entropy(vec: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution vector.
    
    Args:
        vec: Probability distribution vector (should sum to 1.0)
    
    Returns:
        Entropy value (0 = concentrated, max = evenly distributed)
    """
    # Only consider positive values
    p = vec[vec > 0]
    if len(p) == 0:
        return 0.0
    
    # Avoid log(0) by using small epsilon
    p = np.clip(p, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(vec: np.ndarray) -> float:
    """
    Compute normalized entropy (0-1 scale).
    
    Normalized by log(k) where k is the number of non-zero elements.
    This gives:
    - 0.0 for 100% concentration (single segment)
    - 1.0 for even distribution across all segments
    
    Args:
        vec: Probability distribution vector
    
    Returns:
        Normalized entropy in [0, 1]
    """
    k = (vec > 0).sum()
    if k <= 1:
        return 0.0
    
    ent = entropy(vec)
    max_ent = np.log(k)
    if max_ent == 0:
        return 0.0
    
    return float(ent / max_ent)


def compute_segment_similarity(
    target_data: Dict,
    candidate_data: Dict,
    vocabulary: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute segment distribution similarity between target and candidate.
    
    Returns:
        Dict with:
            - segment_similarity: float [0, 1] (cosine similarity)
            - concentration_penalty: float [0, 1] (1 - normalized_entropy)
            - normalized_entropy: float [0, 1] (entropy of candidate's distribution)
    
    Args:
        target_data: Target company dict
        candidate_data: Candidate company dict
        vocabulary: Optional segment vocabulary (will be built if not provided)
    """
    target_dist = extract_segment_distribution(target_data)
    candidate_dist = extract_segment_distribution(candidate_data)
    
    # Build vocabulary if not provided
    if vocabulary is None:
        all_segments = set(target_dist.keys())
        all_segments.update(candidate_dist.keys())
        vocabulary = sorted(all_segments)
    
    if not vocabulary:
        return {
            'segment_similarity': 0.0,
            'concentration_penalty': 1.0,
            'normalized_entropy': 0.0
        }
    
    # Convert to vectors
    target_vec = segment_vector(target_dist, vocabulary)
    candidate_vec = segment_vector(candidate_dist, vocabulary)
    
    # Compute cosine similarity
    seg_sim = cosine_similarity(target_vec, candidate_vec)
    
    # Compute entropy-based concentration penalty
    # High concentration (single segment) → high penalty
    # Even distribution → low penalty
    cand_entropy_norm = normalized_entropy(candidate_vec)
    concentration_penalty = 1.0 - cand_entropy_norm
    
    return {
        'segment_similarity': seg_sim,
        'concentration_penalty': concentration_penalty,
        'normalized_entropy': cand_entropy_norm
    }


def compute_segment_s_score(
    target_data: Dict,
    candidate_data: Dict,
    vocabulary: Optional[List[str]] = None,
    lambda_entropy: float = 0.4,
    diversification_threshold: float = 0.3
) -> Dict[str, float]:
    """
    Compute S (segmentation/customer) feature using segment distribution similarity.
    
    Segment similarity (sim_cosine) is the PRIMARY score - it's a first-class feature.
    Entropy penalty is applied ONLY when target is diversified (multi-segment).
    
    Logic:
    - If target is diversified (entropy >= threshold): Apply entropy penalty to penalize
      single-segment candidates that don't match the target's portfolio diversity.
    - If target is concentrated (entropy < threshold): No entropy penalty - pure segment
      similarity is sufficient (single-segment targets match single-segment candidates).
    
    Formula when target is diversified:
        S = sim_cosine * (1 - λ * penalty_entropy)
    Formula when target is concentrated:
        S = sim_cosine  (no penalty)
    
    where penalty_entropy = |H_target - H_candidate| (only applied if target is diversified)
    
    Args:
        target_data: Target company dict
        candidate_data: Candidate company dict
        vocabulary: Optional segment vocabulary (will be built if not provided)
        lambda_entropy: Tuning parameter λ (default 0.4) - only used when target is diversified
        diversification_threshold: Entropy threshold to consider target "diversified" (default 0.3)
    
    Returns:
        Dict with:
            - S: float [0, 1] (final segment score - primarily sim_cosine, with conditional entropy penalty)
            - sim_cosine: float [0, 1] (cosine similarity of segment distributions - PRIMARY SCORE)
            - entropy_target: float [0, 1] (normalized entropy of target)
            - entropy_candidate: float [0, 1] (normalized entropy of candidate)
            - penalty_entropy: float [0, 1] (absolute difference in entropy, 0 if target not diversified)
            - is_target_diversified: bool (True if target entropy >= threshold)
            - adjusted_seg_score: float [0, 1] (same as S, for clarity)
            - segment_mix_target: Dict (target's segment distribution)
            - segment_mix_candidate: Dict (candidate's segment distribution)
    """
    target_dist = extract_segment_distribution(target_data)
    candidate_dist = extract_segment_distribution(candidate_data)
    
    # Build vocabulary if not provided
    if vocabulary is None:
        all_segments = set(target_dist.keys())
        all_segments.update(candidate_dist.keys())
        vocabulary = sorted(all_segments)
    
    if not vocabulary:
        return {
            'S': 0.0,
            'sim_cosine': 0.0,
            'entropy_target': 0.0,
            'entropy_candidate': 0.0,
            'penalty_entropy': 0.0,
            'is_target_diversified': False,
            'adjusted_seg_score': 0.0,
            'segment_mix_target': target_dist,
            'segment_mix_candidate': candidate_dist
        }
    
    # Convert to vectors
    target_vec = segment_vector(target_dist, vocabulary)
    candidate_vec = segment_vector(candidate_dist, vocabulary)
    
    # Compute cosine similarity (portfolio overlap) - THIS IS THE PRIMARY SCORE
    sim_cosine = cosine_similarity(target_vec, candidate_vec)
    
    # Compute normalized entropy for both
    entropy_target = normalized_entropy(target_vec)
    entropy_candidate = normalized_entropy(candidate_vec)
    
    # Determine if target is diversified (multi-segment)
    is_target_diversified = entropy_target >= diversification_threshold
    
    # Apply entropy penalty ONLY when target is diversified
    # When target is concentrated (single-segment), no penalty needed
    if is_target_diversified:
        # Target is diversified - penalize candidates that are too concentrated
        # Penalty increases as candidate's concentration differs from target's
        penalty_entropy = abs(entropy_target - entropy_candidate)
        # Final score: cosine similarity discounted by entropy mismatch
        S = sim_cosine * (1.0 - lambda_entropy * penalty_entropy)
        S = max(0.0, min(1.0, S))  # Clip to [0, 1]
    else:
        # Target is concentrated (single-segment) - no entropy penalty
        # Pure segment similarity is sufficient
        penalty_entropy = 0.0
        S = sim_cosine
    
    return {
        'S': S,
        'sim_cosine': sim_cosine,  # PRIMARY SCORE - segment similarity
        'entropy_target': entropy_target,
        'entropy_candidate': entropy_candidate,
        'penalty_entropy': penalty_entropy,
        'is_target_diversified': is_target_diversified,
        'adjusted_seg_score': S,  # Same as S
        'segment_mix_target': target_dist,
        'segment_mix_candidate': candidate_dist
    }

