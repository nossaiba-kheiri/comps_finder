"""
customer_overlap.py: Customer overlap scoring (C), using NLP embeddings for semantic similarity.
Enhanced version with semantic embeddings (like P feature) and fallback to substring matching.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import embedding function (fallback-safe)
try:
    import sys
    import os
    # Add parent directory to path to import from universe module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
    from universe.embeddings_index import get_cached_embedding
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    get_cached_embedding = None


def score_customer_overlap(target_customers, candidate_customer_segment, taxonomy=None, run_with_openai=False, similarity_threshold=0.6):
    """
    Compute customer overlap score C using semantic embeddings (like P feature).
    Falls back to substring matching if embeddings unavailable.
    
    Returns score in [0, 1] and hit count.
    
    Args:
        target_customers: List of target customer strings
        candidate_customer_segment: List of candidate customer segment strings
        taxonomy: (Deprecated - not used)
        run_with_openai: Whether to use OpenAI embeddings (default: False, uses cached/dummy)
        similarity_threshold: Cosine similarity threshold for matching (default: 0.6)
    
    Returns:
        Tuple of (score: float, hits: int)
    """
    if not target_customers or not candidate_customer_segment:
        return 0.0, 0
    
    # Normalize inputs to list of strings
    target_lower = []
    for c in target_customers:
        if isinstance(c, str):
            target_lower.append(c.lower().strip())
        else:
            target_lower.append(str(c).lower().strip())
    
    candidate_lower = []
    if isinstance(candidate_customer_segment, list):
        for cs in candidate_customer_segment:
            if isinstance(cs, str):
                candidate_lower.append(cs.lower().strip())
            else:
                candidate_lower.append(str(cs).lower().strip())
    else:
        candidate_lower = [str(candidate_customer_segment).lower().strip()]
    
    # Remove empty strings
    target_lower = [tc for tc in target_lower if tc]
    candidate_lower = [cc for cc in candidate_lower if cc]
    
    if not target_lower or not candidate_lower:
        return 0.0, 0
    
    hits = 0
    hit_customers = []
    
    # Use NLP embeddings for semantic similarity if available, otherwise fall back to substring matching
    use_embeddings = EMBEDDINGS_AVAILABLE and run_with_openai
    
    if use_embeddings:
        # Method 1: NLP-based semantic similarity using embeddings
        try:
            # Check cache first, collect texts that need embedding
            target_embeddings_dict = {}
            candidate_embeddings_dict = {}
            target_texts_needing_embed = []
            candidate_texts_needing_embed = []
            
            # Check target customers cache
            for tc in target_lower:
                try:
                    from universe.embeddings_index import embedding_cache_path
                    cache_path = embedding_cache_path(tc)
                    if os.path.isfile(cache_path):
                        emb = np.load(cache_path)
                        target_embeddings_dict[tc] = emb
                    else:
                        target_texts_needing_embed.append(tc)
                except Exception:
                    target_texts_needing_embed.append(tc)
            
            # Check candidate segments cache
            for cs in candidate_lower:
                try:
                    from universe.embeddings_index import embedding_cache_path
                    cache_path = embedding_cache_path(cs)
                    if os.path.isfile(cache_path):
                        emb = np.load(cache_path)
                        candidate_embeddings_dict[cs] = emb
                    else:
                        candidate_texts_needing_embed.append(cs)
                except Exception:
                    candidate_texts_needing_embed.append(cs)
            
            # Batch embed texts that are not in cache (if OpenAI is enabled)
            if run_with_openai and (target_texts_needing_embed or candidate_texts_needing_embed):
                try:
                    from universe.embeddings_index import embed_openai_batch
                    import os
                    api_key = os.getenv("OPENAI_API_KEY")
                    
                    # Batch embed all texts that need embedding
                    all_texts_to_embed = target_texts_needing_embed + candidate_texts_needing_embed
                    if all_texts_to_embed:
                        batch_embeddings = embed_openai_batch(all_texts_to_embed, api_key=api_key)
                        
                        # Map embeddings back to target customers (preserve order)
                        target_start_idx = 0
                        for tc in target_texts_needing_embed:
                            idx_in_all = target_texts_needing_embed.index(tc) + target_start_idx
                            if idx_in_all < len(batch_embeddings):
                                target_embeddings_dict[tc] = batch_embeddings[idx_in_all]
                        target_start_idx = len(target_texts_needing_embed)
                        
                        # Map embeddings back to candidate segments
                        for cs in candidate_texts_needing_embed:
                            idx_in_all = target_start_idx + candidate_texts_needing_embed.index(cs)
                            if idx_in_all < len(batch_embeddings):
                                candidate_embeddings_dict[cs] = batch_embeddings[idx_in_all]
                except Exception as e:
                    # If batch embedding fails, fall back to individual calls with error handling
                    import warnings
                    warnings.warn(f"Batch embedding failed for customer overlap, using individual calls: {e}")
                    for tc in target_texts_needing_embed:
                        try:
                            emb = get_cached_embedding(tc, run_with_openai=run_with_openai)
                            if emb is not None:
                                target_embeddings_dict[tc] = emb
                        except Exception as embed_error:
                            # Skip this embedding if it fails (connection error, API error, etc.)
                            warnings.warn(f"Failed to get embedding for '{tc[:50]}...': {embed_error}")
                            continue
                    for cs in candidate_texts_needing_embed:
                        try:
                            emb = get_cached_embedding(cs, run_with_openai=run_with_openai)
                            if emb is not None:
                                candidate_embeddings_dict[cs] = emb
                        except Exception as embed_error:
                            # Skip this embedding if it fails (connection error, API error, etc.)
                            warnings.warn(f"Failed to get embedding for '{cs[:50]}...': {embed_error}")
                            continue
            
            # Convert dictionaries to lists
            target_embeddings = [(tc, target_embeddings_dict[tc]) for tc in target_lower if tc in target_embeddings_dict]
            candidate_embeddings = [(cs, candidate_embeddings_dict[cs]) for cs in candidate_lower if cs in candidate_embeddings_dict]
            
            # Compute cosine similarity for each target-candidate pair
            if target_embeddings and candidate_embeddings:
                total_similarity = 0.0
                best_matches = []
                
                for tc, tc_emb in target_embeddings:
                    best_match = None
                    best_similarity = 0.0
                    
                    for cs, cs_emb in candidate_embeddings:
                        # Compute cosine similarity
                        similarity = cosine_similarity(
                            tc_emb.reshape(1, -1),
                            cs_emb.reshape(1, -1)
                        )[0][0]
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (cs, similarity)
                    
                    # If similarity exceeds threshold, consider it a match
                    if best_match and best_similarity >= similarity_threshold:
                        hits += 1
                        hit_customers.append(tc)
                        total_similarity += best_similarity
                        best_matches.append(best_similarity)
                
                # If we have matches, compute average similarity score
                if hits > 0:
                    avg_similarity = total_similarity / hits
                    # Score is weighted by both hit ratio and average similarity
                    hit_ratio = hits / max(len(target_lower), len(candidate_lower))
                    score = (hit_ratio * 0.5) + (avg_similarity * 0.5)
                    return min(score, 1.0), hits
            
            # If embeddings were used but no matches found, fall back to substring matching
            if hits == 0 and len(target_embeddings) > 0 and len(candidate_embeddings) > 0:
                # Reset for substring matching fallback
                hits = 0
                hit_customers = []
        except Exception as e:
            # If embedding computation fails, fall back to substring matching
            import warnings
            warnings.warn(f"Embedding computation failed for customer overlap, using substring matching: {e}")
            hits = 0
            hit_customers = []
    
    # Method 2: Fallback to substring matching (original method)
    for tc in target_lower:
        for cs in candidate_lower:
            if tc in cs or cs in tc:
                hits += 1
                hit_customers.append(tc)
                break
    
    # Normalize: hits / max possible matches
    # Max possible = min(target customers, candidate segments)
    max_possible = min(len(target_lower), len(candidate_lower))
    score = min(hits / max_possible, 1.0) if max_possible > 0 else 0.0
    
    return score, hits


def score_customer_overlap_with_segment_mix(target_data, candidate_data, segment_s_result=None):
    """
    Compute customer overlap using segment distribution similarity (from S feature).
    This leverages the segment_mix cosine similarity already computed for S feature.
    
    Args:
        target_data: Target company dict
        candidate_data: Candidate company dict
        segment_s_result: Optional precomputed segment_s_score result (to avoid recomputation)
    
    Returns:
        Tuple of (score: float, hits: int)
        Returns (0.0, 0) if segment_mix not available
    """
    # If segment_s_result is provided, use its similarity score
    if segment_s_result and isinstance(segment_s_result, dict):
        sim_cosine = segment_s_result.get('sim_cosine', 0.0)
        segment_mix_target = segment_s_result.get('segment_mix_target', {})
        segment_mix_candidate = segment_s_result.get('segment_mix_candidate', {})
        
        # Use cosine similarity as the customer overlap score
        # Count number of overlapping segment keys as "hits"
        if segment_mix_target and segment_mix_candidate:
            target_segments = set(segment_mix_target.keys())
            candidate_segments = set(segment_mix_candidate.keys())
            hits = len(target_segments & candidate_segments)
            return float(sim_cosine), hits
    
    # Otherwise, try to extract segment_mix directly
    from features.segment_distribution import extract_segment_distribution
    
    target_dist = extract_segment_distribution(target_data)
    candidate_dist = extract_segment_distribution(candidate_data)
    
    if not target_dist or not candidate_dist:
        return 0.0, 0
    
    # Build vocabulary and compute cosine similarity
    all_segments = set(target_dist.keys())
    all_segments.update(candidate_dist.keys())
    vocabulary = sorted(all_segments)
    
    if not vocabulary:
        return 0.0, 0
    
    from features.segment_distribution import segment_vector, cosine_similarity
    
    target_vec = segment_vector(target_dist, vocabulary)
    candidate_vec = segment_vector(candidate_dist, vocabulary)
    
    # Compute cosine similarity
    sim_cosine = cosine_similarity(target_vec, candidate_vec)
    
    # Count overlapping segments as hits
    target_segments = set(target_dist.keys())
    candidate_segments = set(candidate_dist.keys())
    hits = len(target_segments & candidate_segments)
    
    return float(sim_cosine), hits


if __name__ == "__main__":
    # Test
    target = ["Banks", "Retailers"]
    candidate = ["banks", "credit unions", "retail stores"]
    score, hits = score_customer_overlap(target, candidate)
    print(f"Score: {score}, Hits: {hits}")
