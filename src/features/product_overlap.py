"""
product_overlap.py: Product overlap scoring (P), using NLP embeddings for semantic similarity.
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

def score_product_overlap(target_products, candidate_business_activity, taxonomy=None, initiatives=None, run_with_openai=False, similarity_threshold=0.7):
    """
    Compute product overlap score P using NLP embeddings for semantic similarity.
    Returns score in [0, 1], hit count, and concept_matches with materiality.
    
    Args:
        target_products: List of target product strings
        candidate_business_activity: List of candidate business activity strings
        taxonomy: (Deprecated - not used)
        initiatives: List of initiatives with materiality_0_1 (optional)
        run_with_openai: Whether to use OpenAI embeddings (default: False, uses cached/dummy)
        similarity_threshold: Cosine similarity threshold for matching (default: 0.7)
    """
    if not target_products or not candidate_business_activity:
        return 0.0, 0, []
    
    # Normalize target products to list of strings
    target_lower = []
    for p in target_products:
        if isinstance(p, str):
            target_lower.append(p.lower().strip())
        else:
            target_lower.append(str(p).lower().strip())
    
    # Normalize candidate business_activity to list of strings
    # Handle both list and string formats (e.g., comma-separated string)
    candidate_lower = []
    if isinstance(candidate_business_activity, list):
        for b in candidate_business_activity:
            if isinstance(b, str):
                # If it's a comma-separated string, split it
                if ',' in b:
                    for item in b.split(','):
                        candidate_lower.append(item.strip().lower())
                else:
                    candidate_lower.append(b.lower().strip())
            else:
                candidate_lower.append(str(b).lower().strip())
    elif isinstance(candidate_business_activity, str):
        # Single string - split by comma if needed
        if ',' in candidate_business_activity:
            for item in candidate_business_activity.split(','):
                candidate_lower.append(item.strip().lower())
        else:
            candidate_lower.append(candidate_business_activity.lower().strip())
    else:
        # Fallback for other types
        candidate_lower = [str(candidate_business_activity).lower().strip()]
    
    # Build materiality map from initiatives
    materiality_map = {}
    if initiatives:
        for initiative in initiatives:
            name = initiative.get('name', '').lower()
            category = initiative.get('category', '')
            materiality = initiative.get('materiality_0_1', 0.1)
            # Map initiative name to materiality
            if category in ['product', 'service']:
                materiality_map[name] = materiality
                # Also map aliases/descriptions
                desc = initiative.get('description', '').lower()
                if desc:
                    materiality_map[desc] = materiality
    
    # Default materiality for main business activities (not in initiatives)
    default_materiality = 1.0
    
    hits = 0
    hit_products = []
    concept_matches = []
    weighted_score = 0.0
    
    # Use NLP embeddings for semantic similarity if available, otherwise fall back to substring matching
    use_embeddings = EMBEDDINGS_AVAILABLE and run_with_openai
    
    if use_embeddings:
        # Method 1: NLP-based semantic similarity using embeddings
        # Compute embeddings for all target products and candidate activities
        target_embeddings = []
        candidate_embeddings = []
        
        try:
            # Batch embedding requests for better performance
            # Get embeddings for target products
            target_texts_to_embed = [tp for tp in target_lower]
            candidate_texts_to_embed = [ca for ca in candidate_lower]
            
            # Check cache first, collect texts that need embedding
            target_embeddings_dict = {}
            candidate_embeddings_dict = {}
            target_texts_needing_embed = []
            candidate_texts_needing_embed = []
            
            # Check target products cache
            import os
            for tp in target_lower:
                # Try to load from cache first (without API call)
                try:
                    from universe.embeddings_index import embedding_cache_path
                    cache_path = embedding_cache_path(tp)
                    if os.path.isfile(cache_path):
                        emb = np.load(cache_path)
                        target_embeddings_dict[tp] = emb
                    else:
                        target_texts_needing_embed.append(tp)
                except Exception:
                    # If cache check fails, add to list for embedding
                    target_texts_needing_embed.append(tp)
            
            # Check candidate activities cache
            for ca in candidate_lower:
                # Try to load from cache first (without API call)
                try:
                    from universe.embeddings_index import embedding_cache_path
                    cache_path = embedding_cache_path(ca)
                    if os.path.isfile(cache_path):
                        emb = np.load(cache_path)
                        candidate_embeddings_dict[ca] = emb
                    else:
                        candidate_texts_needing_embed.append(ca)
                except Exception:
                    # If cache check fails, add to list for embedding
                    candidate_texts_needing_embed.append(ca)
            
            # Batch embed texts that are not in cache (if OpenAI is enabled)
            if run_with_openai and (target_texts_needing_embed or candidate_texts_needing_embed):
                try:
                    # Import batch embedding function
                    from universe.embeddings_index import embed_openai_batch
                    import os
                    api_key = os.getenv("OPENAI_API_KEY")
                    
                    # Batch embed all texts that need embedding
                    all_texts_to_embed = target_texts_needing_embed + candidate_texts_needing_embed
                    if all_texts_to_embed:
                        batch_embeddings = embed_openai_batch(all_texts_to_embed, api_key=api_key)
                        
                        # Map embeddings back to target products (preserve order)
                        target_start_idx = 0
                        for tp in target_texts_needing_embed:
                            idx_in_all = target_texts_needing_embed.index(tp) + target_start_idx
                            if idx_in_all < len(batch_embeddings):
                                target_embeddings_dict[tp] = batch_embeddings[idx_in_all]
                        target_start_idx = len(target_texts_needing_embed)
                        
                        # Map embeddings back to candidate activities
                        for ca in candidate_texts_needing_embed:
                            idx_in_all = target_start_idx + candidate_texts_needing_embed.index(ca)
                            if idx_in_all < len(batch_embeddings):
                                candidate_embeddings_dict[ca] = batch_embeddings[idx_in_all]
                except Exception as e:
                    # If batch embedding fails, fall back to individual calls
                    import warnings
                    warnings.warn(f"Batch embedding failed, using individual calls: {e}")
                    for tp in target_texts_needing_embed:
                        emb = get_cached_embedding(tp, run_with_openai=run_with_openai)
                        if emb is not None:
                            target_embeddings_dict[tp] = emb
                    for ca in candidate_texts_needing_embed:
                        emb = get_cached_embedding(ca, run_with_openai=run_with_openai)
                        if emb is not None:
                            candidate_embeddings_dict[ca] = emb
            
            # Convert dictionaries to lists
            target_embeddings = [(tp, target_embeddings_dict[tp]) for tp in target_lower if tp in target_embeddings_dict]
            candidate_embeddings = [(ca, candidate_embeddings_dict[ca]) for ca in candidate_lower if ca in candidate_embeddings_dict]
            
            # Compute cosine similarity for each target-candidate pair
            for tp, tp_emb in target_embeddings:
                best_match = None
                best_similarity = 0.0
                
                for ca, ca_emb in candidate_embeddings:
                    # Compute cosine similarity
                    similarity = cosine_similarity(
                        tp_emb.reshape(1, -1),
                        ca_emb.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (ca, similarity)
                
                # If similarity exceeds threshold, consider it a match
                if best_match and best_similarity >= similarity_threshold:
                    ca_matched, similarity_score = best_match
                    
                    # Get materiality for this match
                    materiality = materiality_map.get(ca_matched, default_materiality)
                    if ca_matched not in materiality_map:
                        materiality = default_materiality
                    
                    hits += 1
                    hit_products.append(tp)
                    weighted_score += materiality * best_similarity  # Weight by similarity
                    concept_matches.append({
                        'concept': ca_matched,
                        'match_type': 'semantic',
                        'materiality_0_1': materiality,
                        'match_strength': float(best_similarity),
                        'similarity': float(best_similarity)
                    })
        
        except Exception as e:
            # If embeddings fail, fall back to substring matching
            use_embeddings = False
    
    if not use_embeddings:
        # Method 2: Fallback to substring and word-level matching
        for tp in target_lower:
            tp_words = set(tp.split())
            matched = False
            matched_ca = None
            
            for ca in candidate_lower:
                ca_words = set(ca.split())
                
                # Substring matching (full phrase)
                if tp in ca or ca in tp:
                    matched = True
                    matched_ca = ca
                    break
                # Word-level matching
                elif len(tp_words) > 0 and len(ca_words) > 0:
                    # Normalize words: split hyphenated words
                    def normalize_words(word_set):
                        normalized = set()
                        for w in word_set:
                            if '-' in w:
                                normalized.update(w.lower().split('-'))
                            else:
                                normalized.add(w.lower())
                        return normalized
                    
                    tp_normalized = normalize_words(tp_words)
                    ca_normalized = normalize_words(ca_words)
                    
                    significant_words = [w for w in tp_normalized if len(w) >= 2]
                    common_words = [w for w in significant_words if w in ca_normalized]
                    
                    if len(significant_words) > 0:
                        min_required = 1 if len(significant_words) <= 2 else max(2, int(len(significant_words) * 0.5))
                        if len(common_words) >= min_required:
                            matched = True
                            matched_ca = ca
                            break
            
            if matched and matched_ca:
                # Get materiality for this match
                materiality = materiality_map.get(matched_ca, default_materiality)
                if matched_ca not in materiality_map:
                    materiality = default_materiality
                
                hits += 1
                hit_products.append(tp)
                weighted_score += materiality
                concept_matches.append({
                    'concept': matched_ca,
                    'match_type': 'substring',
                    'materiality_0_1': materiality,
                    'match_strength': 1.0
                })
    
    # Normalize: weighted score / max possible score
    # Max possible = min(target products, candidate activities) × max materiality
    # Use candidate_lower length (after splitting comma-separated strings)
    max_possible_hits = min(len(target_lower), len(candidate_lower))
    # Use max materiality from initiatives, or default 1.0
    max_materiality = max(materiality_map.values()) if materiality_map else default_materiality
    # For embeddings, max_score is adjusted by similarity threshold
    if use_embeddings:
        max_score = max_possible_hits * max_materiality * 1.0 if max_possible_hits > 0 else 1.0
    else:
        max_score = max_possible_hits * max_materiality if max_possible_hits > 0 else 1.0
    
    # Normalize score, ensuring it's in [0, 1]
    score = min(weighted_score / max_score, 1.0) if max_score > 0 else 0.0
    
    # Sort concept_matches by materiality × match_strength (descending)
    concept_matches.sort(key=lambda x: x['materiality_0_1'] * x['match_strength'], reverse=True)
    
    return score, hits, concept_matches


if __name__ == "__main__":
    # Test
    target = ["Payment Processing", "Transaction APIs"]
    candidate = ["payment services", "financial transactions", "API solutions"]
    score, hits = score_product_overlap(target, candidate)
    print(f"Score: {score}, Hits: {hits}")
