"""
rank_key_semantic.py: Compute sophisticated rank_key using semantic, segment mix, and archetype similarity.

This replaces the crude keyword-based rank_key with a more accurate, generic approach that:
- Uses embeddings for product similarity (P_fast)
- Uses segment distribution similarity (C_mix)
- Uses business model/archetype similarity (M_fast)
- Applies entropy penalty for diversification mismatch

Generic - works for any industry, no hardcoding.
"""
import numpy as np
from typing import Dict, Optional, Tuple
import sys
import os

# Import embeddings
sys.path.insert(0, os.path.dirname(__file__))
try:
    from embeddings_index import get_cached_embedding
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    def get_cached_embedding(text, run_with_openai=False, model='text-embedding-3-large'):
        return None

# Import segment distribution utilities
features_path = os.path.join(os.path.dirname(__file__), '../features')
sys.path.insert(0, features_path)
try:
    from segment_distribution import (
        extract_segment_distribution,
        cosine_similarity as seg_cosine_similarity,
        normalized_entropy,
        segment_vector,
        build_segment_vocabulary
    )
    SEGMENT_DIST_AVAILABLE = True
except ImportError:
    SEGMENT_DIST_AVAILABLE = False
    def extract_segment_distribution(*args, **kwargs):
        return {}
    def seg_cosine_similarity(*args, **kwargs):
        return 0.5
    def normalized_entropy(*args, **kwargs):
        return 0.0
    def segment_vector(*args, **kwargs):
        return np.array([])
    def build_segment_vocabulary(*args, **kwargs):
        return []

# Import model similarity utilities
try:
    from model_similarity import (
        archetype_vector,
        compute_model_similarities,
        cosine_similarity as model_cosine_similarity
    )
    MODEL_SIM_AVAILABLE = True
except ImportError:
    MODEL_SIM_AVAILABLE = False
    def archetype_vector(*args, **kwargs):
        return np.array([0.25, 0.25, 0.25, 0.25])
    def compute_model_similarities(*args, **kwargs):
        return {'archetype_similarity': 0.5}
    def model_cosine_similarity(*args, **kwargs):
        return 0.5


def compute_product_similarity_fast(
    target_products: list,
    candidate_summary: str,
    run_with_openai: bool = False
) -> float:
    """
    Compute fast product similarity using embeddings.
    
    Args:
        target_products: List of target product/service keywords
        candidate_summary: Candidate company summary text
        run_with_openai: Whether to use OpenAI embeddings
    
    Returns:
        float [0, 1]: Product similarity score
    """
    if not target_products or not candidate_summary:
        return 0.0
    
    if not EMBEDDINGS_AVAILABLE:
        # Fallback: simple keyword overlap
        candidate_lower = str(candidate_summary).lower()
        hits = sum(1 for p in target_products if str(p).lower() in candidate_lower)
        return min(hits / max(len(target_products), 1), 1.0)
    
    try:
        # Embed target products (ensure all are strings)
        target_text = ' '.join([str(p) for p in target_products])
        target_emb = get_cached_embedding(target_text, run_with_openai=run_with_openai)
        if target_emb is None:
            return 0.0
        
        # Embed candidate summary (truncate to reasonable length)
        candidate_text = candidate_summary[:2000]  # Limit for efficiency
        candidate_emb = get_cached_embedding(candidate_text, run_with_openai=run_with_openai)
        if candidate_emb is None:
            return 0.0
        
        # Compute cosine similarity
        target_vec = np.array(target_emb)
        candidate_vec = np.array(candidate_emb)
        
        # Normalize
        norm_target = np.linalg.norm(target_vec)
        norm_candidate = np.linalg.norm(candidate_vec)
        if norm_target == 0 or norm_candidate == 0:
            return 0.0
        
        cosine = np.dot(target_vec, candidate_vec) / (norm_target * norm_candidate)
        
        # Map from [-1, 1] to [0, 1]
        return float((cosine + 1.0) / 2.0)
    except Exception:
        # Fallback on error
        return 0.0


def compute_segment_mix_similarity_fast(
    target_data: Dict,
    candidate_data: Dict,
    vocabulary: Optional[list] = None
) -> Tuple[float, float, float]:
    """
    Compute segment mix similarity and entropy metrics.
    
    Args:
        target_data: Target company dict (with segment_mix, customer_segment, etc.)
        candidate_data: Candidate company dict (with summary, customer_segment, etc.)
        vocabulary: Optional segment vocabulary (will be built if not provided)
    
    Returns:
        Tuple of:
        - C_mix: float [0, 1] (cosine similarity of segment distributions)
        - entropy_target: float [0, 1] (normalized entropy of target)
        - entropy_candidate: float [0, 1] (normalized entropy of candidate)
    """
    if not SEGMENT_DIST_AVAILABLE:
        return 0.5, 0.0, 0.0
    
    target_dist = extract_segment_distribution(target_data)
    candidate_dist = extract_segment_distribution(candidate_data)
    
    # If no segments found, return neutral scores
    if not target_dist and not candidate_dist:
        return 0.5, 0.0, 0.0
    
    # CRITICAL FIX: If candidate has no real segment_mix (only inferred from own industry/sector),
    # return neutral C_mix instead of 0.0 to avoid penalizing companies without segment data.
    # At preliminary ranking stage, we don't have LLM-extracted segment_mix yet.
    # Check if candidate_dist looks like it was inferred from own industry (not real customer segments)
    candidate_segment = candidate_data.get('customer_segment', [])
    has_real_segment_mix = candidate_data.get('segment_mix', {}) and len(candidate_data.get('segment_mix', {})) > 0
    
    # If candidate_dist was inferred from customer_segment that looks like own industry/sector,
    # and we don't have real segment_mix, treat it as unreliable and return neutral
    if not has_real_segment_mix and candidate_dist:
        # Check if candidate_segment contains industry/sector terms (unreliable)
        candidate_industry = str(candidate_data.get('industry', '')).lower()
        candidate_sector = str(candidate_data.get('sector', '')).lower()
        
        # If customer_segment matches own industry/sector, it's unreliable
        segment_is_own_industry = any(
            candidate_industry in str(seg).lower() or 
            candidate_sector in str(seg).lower() or
            str(seg).lower() in candidate_industry or
            str(seg).lower() in candidate_sector
            for seg in candidate_segment
        )
        
        if segment_is_own_industry:
            # Unreliable segment data - use optimistic neutral C_mix (0.7 instead of 0.5)
            # This gives companies the benefit of the doubt when we don't have segment data yet
            # Also return matching entropy to avoid entropy penalty
            if target_dist:
                target_vec = segment_vector(target_dist, list(target_dist.keys()))
                entropy_target = normalized_entropy(target_vec)
                # Return matching entropy to avoid penalty (we don't know candidate's real entropy)
                entropy_candidate = entropy_target
            else:
                entropy_target = 0.0
                entropy_candidate = 0.0
            return 0.7, entropy_target, entropy_candidate  # Optimistic neutral C_mix (0.7), matching entropies to avoid penalty
    
    # Build vocabulary if not provided
    if vocabulary is None:
        all_segments = set(target_dist.keys())
        all_segments.update(candidate_dist.keys())
        vocabulary = sorted(all_segments)
    
    if not vocabulary:
        return 0.5, 0.0, 0.0
    
    # Convert to vectors
    target_vec = segment_vector(target_dist, vocabulary)
    candidate_vec = segment_vector(candidate_dist, vocabulary)
    
    # Compute cosine similarity
    C_mix = seg_cosine_similarity(target_vec, candidate_vec)
    
    # Compute normalized entropy
    entropy_target = normalized_entropy(target_vec)
    entropy_candidate = normalized_entropy(candidate_vec)
    
    return C_mix, entropy_target, entropy_candidate


def compute_industry_match_bonus(
    target_data: Dict,
    candidate_industry: str,
    candidate_sector: str = ''
) -> float:
    """
    Compute industry match bonus to prioritize companies in same/related industries.
    
    Returns:
        - 1.0: Exact match with primary_industry_classification or similar_industries
        - 0.7: Related industry (contains keywords from similar_industries)
        - 0.3: Partial match (some overlap)
        - 0.0: No match
    """
    if not candidate_industry:
        return 0.0
    
    candidate_industry_lower = candidate_industry.lower()
    candidate_sector_lower = candidate_sector.lower() if candidate_sector else ''
    
    # Get target's main industry
    primary_industry = str(target_data.get('primary_industry_classification', '')).lower()
    similar_industries = target_data.get('similar_industries', [])
    if not isinstance(similar_industries, list):
        similar_industries = []
    similar_industries_lower = [str(ind).lower() for ind in similar_industries]
    
    # Check exact match with primary industry
    # Be strict: require substantial overlap, not just one word
    if primary_industry:
        # Extract key terms from primary industry (e.g., "Research and Consulting Services" -> ["research", "consulting", "services"])
        primary_terms = set([t.strip() for t in primary_industry.replace(',', ' ').replace('&', ' ').split() if len(t.strip()) > 3])
        if primary_terms:
            matches = sum(1 for term in primary_terms if term in candidate_industry_lower or term in candidate_sector_lower)
            # Require at least 2 matches AND the candidate industry should contain a substantial portion
            if matches >= 2 and len(primary_terms) >= 2:
                # Check if candidate industry is a subset or very similar (not just "Specialty Business Services" matching "Business Services")
                # Reject if candidate has extra words that change meaning (e.g., "Specialty" prefix)
                if 'specialty' in candidate_industry_lower and 'specialty' not in primary_industry.lower():
                    return 0.7  # Downgrade "Specialty X" matching "X"
                return 1.0
    
    # Check exact match with similar industries - HIERARCHICAL: first industry gets highest weight
    # Iterate through similar_industries in order (first = most important)
    for idx, similar_ind in enumerate(similar_industries_lower):
        if not similar_ind:
            continue
        
        # Calculate hierarchical bonus: first industry = 1.0, decreases by 0.1 per position
        # Position 0 (first) = 1.0, position 1 = 0.9, position 2 = 0.8, etc.
        position_bonus = max(0.5, 1.0 - (idx * 0.1))  # Minimum 0.5 even for later positions
        
        # Exact match or candidate contains the full similar industry
        if similar_ind in candidate_industry_lower:
            # But reject if candidate has "Specialty" prefix and similar_ind doesn't
            if 'specialty' in candidate_industry_lower and 'specialty' not in similar_ind:
                return max(0.5, position_bonus - 0.2)  # Downgrade but still consider position
            # Boost for exact matches (e.g., "Information Technology Services" = "Information Technology Services")
            if candidate_industry_lower == similar_ind:
                return position_bonus  # Return hierarchical bonus based on position
            return position_bonus
        # Check if candidate industry is contained in similar industry (reverse direction)
        if len(similar_ind) > 5 and similar_ind in candidate_industry_lower:
            # But reject if candidate has extra qualifiers
            if 'specialty' in candidate_industry_lower and 'specialty' not in similar_ind:
                return max(0.5, position_bonus - 0.2)  # Downgrade but still consider position
            return position_bonus
    
    # Check if candidate industry contains keywords from similar industries
    # Extract key terms from similar industries (e.g., "Consulting Services" -> ["consulting", "services"])
    key_terms = set()
    for similar_ind in similar_industries_lower:
        if similar_ind:
            # Split by common separators and add individual words
            terms = similar_ind.replace(',', ' ').replace('&', ' ').split()
            key_terms.update([t.strip() for t in terms if len(t.strip()) > 3])  # Only meaningful words
    
    # Check if candidate industry contains any key terms
    if key_terms:
        matches = sum(1 for term in key_terms if term in candidate_industry_lower or term in candidate_sector_lower)
        if matches >= 2:  # Multiple keyword matches = related industry
            return 0.7
        elif matches >= 1:  # Single keyword match = partial match
            return 0.3
    
    return 0.0


def compute_model_similarity_fast(
    target_data: Dict,
    candidate_summary: str,
    candidate_industry: str = ''
) -> float:
    """
    Compute fast business model/archetype similarity.
    
    Uses target's revenue_archetypes if available, otherwise infers from summary.
    
    Args:
        target_data: Target company dict (with revenue_archetypes, business_model_type, etc.)
        candidate_summary: Candidate company summary text
        candidate_industry: Candidate industry (for fallback inference)
    
    Returns:
        float [0, 1]: Model similarity score
    """
    if not MODEL_SIM_AVAILABLE:
        return 0.5
    
    # Try to get target archetypes
    target_archetypes = target_data.get('revenue_archetypes', {})
    
    if target_archetypes:
        # We have target archetypes - need to infer candidate archetypes from summary
        # For now, use a simple heuristic based on summary keywords
        # This is a fast approximation - full LLM extraction will refine this later
        
        candidate_lower = str(candidate_summary).lower() + ' ' + str(candidate_industry).lower()
        
        # Infer candidate archetype distribution from keywords
        # This is a heuristic - full extraction will be more accurate
        cand_archetypes = {}
        
        # unit_of_work: consulting, services, staffing, implementation
        if any(kw in candidate_lower for kw in ['consulting', 'services', 'staffing', 'implementation', 'advisory', 'professional services']):
            cand_archetypes['unit_of_work'] = 0.7
        else:
            cand_archetypes['unit_of_work'] = 0.1
        
        # access_capability: software, platform, saas, subscription
        if any(kw in candidate_lower for kw in ['software', 'platform', 'saas', 'subscription', 'cloud', 'application']):
            cand_archetypes['access_capability'] = 0.7
        else:
            cand_archetypes['access_capability'] = 0.1
        
        # performance_outcome: outcomes, results, performance, bpo
        if any(kw in candidate_lower for kw in ['outcomes', 'results', 'performance', 'bpo', 'business process']):
            cand_archetypes['performance_outcome'] = 0.5
        else:
            cand_archetypes['performance_outcome'] = 0.1
        
        # intermediation: marketplace, broker, exchange, transaction
        if any(kw in candidate_lower for kw in ['marketplace', 'broker', 'exchange', 'transaction', 'platform']):
            cand_archetypes['intermediation'] = 0.5
        else:
            cand_archetypes['intermediation'] = 0.1
        
        # Normalize
        total = sum(cand_archetypes.values())
        if total > 0:
            cand_archetypes = {k: v / total for k, v in cand_archetypes.items()}
        
        # Compute similarity
        target_vec = archetype_vector(target_archetypes)
        candidate_vec = archetype_vector(cand_archetypes)
        
        M_fast = model_cosine_similarity(target_vec, candidate_vec)
        return M_fast
    else:
        # No target archetypes - return neutral
        return 0.5


def compute_rank_key_semantic(
    target: Dict,
    candidate_data: Dict,
    config: Optional[Dict] = None,
    vocabulary: Optional[list] = None,
    run_with_openai: bool = False
) -> Dict[str, float]:
    """
    Compute sophisticated rank_key using semantic, segment mix, and archetype similarity.
    
    Formula:
        rank_key = w_prod * P_fast + w_seg * C_mix + w_model * M_fast - w_entropy * entropy_penalty
    
    Args:
        target: Target company dict
        candidate_data: Candidate company dict (with ticker, summary, industry, etc.)
        config: Optional config dict with stage1_rank weights
        vocabulary: Optional segment vocabulary
        run_with_openai: Whether to use OpenAI embeddings
    
    Returns:
        Dict with:
            - rank_key: float [0, 1] (final rank_key score)
            - P_fast: float [0, 1] (product similarity)
            - C_mix: float [0, 1] (segment mix similarity)
            - M_fast: float [0, 1] (model similarity)
            - entropy_penalty: float [0, 1] (entropy mismatch penalty)
            - entropy_target: float [0, 1] (target entropy)
            - entropy_candidate: float [0, 1] (candidate entropy)
    """
    # Load config weights (with defaults)
    if config is None:
        config = {}
    
    stage1_config = config.get('stage1_rank', {})
    w_prod = stage1_config.get('w_product', 0.40)
    w_seg = stage1_config.get('w_segments', 0.40)
    w_model = stage1_config.get('w_model', 0.20)
    w_entropy = stage1_config.get('w_entropy_penalty', 0.15)
    
    # ONLY use business_activity - no products field, no fallback
    # Compare business_activity to company descriptions (summary) in universe
    target_products_raw = target.get('business_activity', [])
    # Ensure all items are strings (handle case where list contains floats or other types)
    target_products = [str(p) for p in target_products_raw] if target_products_raw else []
    
    # Get candidate summary
    candidate_summary = str(candidate_data.get('summary', ''))
    candidate_industry = str(candidate_data.get('industry', ''))
    
    # 1. Compute P_fast (product similarity)
    P_fast = compute_product_similarity_fast(
        target_products,
        candidate_summary,
        run_with_openai=run_with_openai
    )
    
    # 2. Compute C_mix (segment mix similarity) and entropy metrics
    C_mix, entropy_target, entropy_candidate = compute_segment_mix_similarity_fast(
        target,
        candidate_data,
        vocabulary=vocabulary
    )
    
    # 3. Compute M_fast (model similarity) - COMPUTED BUT NOT USED IN RANK_KEY
    # M_fast was giving high scores (0.993) to unrelated companies, causing bias
    M_fast = compute_model_similarity_fast(
        target,
        candidate_summary,
        candidate_industry
    )
    
    # 4. Compute entropy penalty
    entropy_penalty = abs(entropy_target - entropy_candidate)
    
    # 5. Compute industry match bonus (prioritize same/related industries)
    # This now returns hierarchical bonus based on position in similar_industries list
    industry_bonus = compute_industry_match_bonus(
        target,
        candidate_industry,
        candidate_data.get('sector', '')
    )
    
    # 6. Compute rank_key (WITHOUT M_fast - removed to avoid bias)
    # PRIORITIZE INDUSTRY MATCH FIRST - companies in same industry should rank higher
    w_industry = stage1_config.get('w_industry_bonus', 0.55)  # Default 55% weight for industry bonus (MAXIMUM PRIORITY)
    # Note: w_model is ignored - M_fast is not used in rank_key calculation
    
    # Additional boost for exact industry matches (hierarchical - first industry gets more)
    # This helps prioritize true industry peers over general companies
    industry_boost = 0.0
    if industry_bonus >= 0.5:  # Any industry match (hierarchical bonus already applied)
        # Check which position in similar_industries list the match is at
        candidate_industry_lower = str(candidate_data.get('industry', '')).lower()
        similar_industries = target.get('similar_industries', [])
        if isinstance(similar_industries, list):
            similar_industries_lower = [str(ind).lower() for ind in similar_industries]
            # Find the position of the match
            for idx, similar_ind in enumerate(similar_industries_lower):
                if candidate_industry_lower == similar_ind or similar_ind in candidate_industry_lower:
                    # Hierarchical boost: first industry gets 0.15, decreases by 0.02 per position
                    industry_boost = max(0.05, 0.15 - (idx * 0.02))  # Minimum 0.05 even for later positions
                    break
    
    # Note: Removed hardcoded "consulting boost" - it was not generic
    # The industry_bonus and P_fast (using business_activity) should be sufficient
    # to prioritize relevant companies without hardcoding specific keywords
    
    rank_key = (
        w_prod * P_fast +
        w_seg * C_mix -
        w_entropy * entropy_penalty +
        w_industry * industry_bonus +
        industry_boost
    )
    
    # Clip to [0, 1]
    rank_key = max(0.0, min(1.0, rank_key))
    
    return {
        'rank_key': rank_key,
        'P_fast': P_fast,
        'C_mix': C_mix,
        'M_fast': M_fast,
        'entropy_penalty': entropy_penalty,
        'entropy_target': entropy_target,
        'entropy_candidate': entropy_candidate,
        'industry_bonus': industry_bonus
    }

