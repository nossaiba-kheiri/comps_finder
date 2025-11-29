"""
generate_candidates.py: Candidate generation from universe via all recall paths (A–D/E).
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
import faiss
from pathlib import Path

# Import new semantic rank_key computation
sys.path.insert(0, os.path.dirname(__file__))
try:
    from rank_key_semantic import compute_rank_key_semantic, build_segment_vocabulary
    SEMANTIC_RANK_AVAILABLE = True
except ImportError:
    SEMANTIC_RANK_AVAILABLE = False
    def compute_rank_key_semantic(*args, **kwargs):
        return {'rank_key': 0.0, 'P_fast': 0.0, 'C_mix': 0.0, 'M_fast': 0.0, 'entropy_penalty': 0.0, 'entropy_target': 0.0, 'entropy_candidate': 0.0}

# Import embeddings utilities
sys.path.insert(0, os.path.dirname(__file__))
try:
    from embeddings_index import preprocess, get_cached_embedding, embedding_cache_path
except ImportError:
    # Fallback if not available
    def preprocess(text):
        import re
        text = (text or "").lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:8000]
    
    # Import from embeddings_index if available
    try:
        from embeddings_index import get_cached_embedding
    except ImportError:
        def get_cached_embedding(text, run_with_openai=False, api_key=None, model='text-embedding-3-large'):
            # Simplified version for candidate generation
            import numpy as np
            np.random.seed(hash(text) % (2**31))
            v = np.random.randn(3072).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

UNIVERSE_PATH = os.path.join(os.path.dirname(__file__), '../../data/universe_us.csv')
FAISS_PATH = os.path.join(os.path.dirname(__file__), '../../data/embeddings/universe_index.faiss')
META_PATH = os.path.join(os.path.dirname(__file__), '../../data/embeddings/universe_meta.parquet')
SEGMENTS_ALIAS_PATH = os.path.join(os.path.dirname(__file__), '../../config/segments_alias.csv')


def load_segments_alias():
    """Load segment aliases for matching."""
    if os.path.exists(SEGMENTS_ALIAS_PATH):
        return pd.read_csv(SEGMENTS_ALIAS_PATH)
    return pd.DataFrame(columns=['alias', 'canonical'])


def compute_keyword_overlap(text, keywords, taxonomy=None):
    """
    Compute keyword overlap score between text and keyword list.
    Uses direct substring matching only (taxonomy deprecated).
    
    Args:
        text: Text to search in
        keywords: List of keywords to search for
        taxonomy: (Deprecated - not used) Taxonomy dict for keyword matching
    """
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    hits = 0
    
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            hits += 1
    
    return min(hits / max(len(keywords), 1), 1.0)


def get_permitted_industries(target_products, taxonomy=None):
    """
    Map target products to all permitted industries via taxonomy.
    (Deprecated - returns empty set, taxonomy matching removed)
    
    Args:
        target_products: List of target product strings
        taxonomy: (Deprecated - not used) Taxonomy dict for industry mapping
    
    Returns:
        Empty set (taxonomy matching removed)
    """
    return set()  # Taxonomy matching removed - return empty set


def generate_candidates(target, config, run_with_openai=False):
    """
    Generate candidates using multiple recall paths:
    A. Cross-sector inclusion: map products to permitted industries, include all tickers in those industries
    B. Ontology keywords (P_kw, C_kw)
    C. Semantic sweep (ANN/FAISS)
    D. Segment alias lift
    E. (Optional) weak signals
    
    Uses sophisticated semantic rank_key for shortlisting (replaces crude keyword-based approach).
    """
    # Load universe and embeddings
    universe_df = pd.read_csv(UNIVERSE_PATH)
    index = faiss.read_index(FAISS_PATH) if os.path.exists(FAISS_PATH) else None
    meta_df = pd.read_parquet(META_PATH) if os.path.exists(META_PATH) else universe_df
    
    # Check if we should use semantic rank_key (default: True)
    use_semantic_rank = config.get('use_semantic_rank_key', True)
    
    # Taxonomy removed - using direct substring matching only
    segments_alias = load_segments_alias()
    
    # Get recall config
    K_total = config.get('recall', {}).get('K_total', 200)
    K_semantic = config.get('recall', {}).get('K_semantic', 80)
    K_keywords = config.get('recall', {}).get('K_keywords', 120)
    K_alias = config.get('recall', {}).get('K_alias', 40)
    epsilon_random = config.get('recall', {}).get('epsilon_random', 10)
    
    # ONLY use business_activity - no products field, no fallback
    # Compare business_activity to company descriptions (summary) in universe
    target_products = target.get('business_activity', [])
    if not target_products:
        # No fallback - if business_activity is missing, use empty list
        target_products = []
    target_customers = target.get('customers', [])
    
    # Path A: Cross-sector inclusion (SEMANTIC MATCHING)
    # CRITICAL: Filter by own industry first (similar_industries), not by customer industries
    # We want companies in the same industry (e.g., consulting firms), not companies in customer industries
    # Uses semantic embeddings instead of substring matching to find related industries
    target_similar_industries = target.get('similar_industries', [])  # Similar own industries for filtering
    target_own_industry = target.get('primary_industry_classification', '').lower()
    
    industry_candidates = set()
    industry_similarity_threshold = 0.65  # Minimum cosine similarity for industry match
    
    # First, try semantic matching by similar_industries (most accurate)
    if target_similar_industries:
        try:
            # Create embeddings for target industries
            target_industry_texts = []
            for similar_ind in target_similar_industries:
                target_industry_texts.append(str(similar_ind).lower().strip())
            
            # Also add primary industry if available
            if target_own_industry:
                target_industry_texts.append(target_own_industry)
            
            # Get unique industry texts
            target_industry_texts = list(set(target_industry_texts))
            
            # Create embeddings for target industries
            target_industry_embeddings = []
            for ind_text in target_industry_texts:
                if ind_text:
                    emb = get_cached_embedding(ind_text, run_with_openai=run_with_openai, model='text-embedding-3-large')
                    emb_np = np.array(emb).astype(np.float32)
                    # Normalize for cosine similarity
                    emb_np = emb_np / np.linalg.norm(emb_np)
                    target_industry_embeddings.append(emb_np)
            
            if target_industry_embeddings:
                # Get unique candidate industries from universe
                candidate_industries = universe_df[['ticker', 'industry', 'sector']].copy()
                candidate_industries['industry_text'] = candidate_industries.apply(
                    lambda row: f"{row['industry']} {row['sector']}".lower().strip() if pd.notna(row['industry']) and pd.notna(row['sector']) else (str(row['industry']).lower().strip() if pd.notna(row['industry']) else ''),
                    axis=1
                )
                
                # Get unique industry texts and their embeddings
                unique_industry_texts = candidate_industries['industry_text'].unique()
                industry_text_to_emb = {}
                
                for ind_text in unique_industry_texts:
                    if ind_text and len(ind_text) > 2:  # Skip empty/short texts
                        try:
                            emb = get_cached_embedding(ind_text, run_with_openai=run_with_openai, model='text-embedding-3-large')
                            emb_np = np.array(emb).astype(np.float32)
                            emb_np = emb_np / np.linalg.norm(emb_np)
                            industry_text_to_emb[ind_text] = emb_np
                        except Exception as e:
                            # Skip if embedding fails
                            continue
                
                # For each candidate, check if its industry matches any target industry semantically
                for _, row in candidate_industries.iterrows():
                    ticker = row['ticker']
                    candidate_ind_text = row['industry_text']
                    
                    if candidate_ind_text in industry_text_to_emb:
                        candidate_emb = industry_text_to_emb[candidate_ind_text]
                        
                        # Check similarity with all target industry embeddings
                        for target_emb in target_industry_embeddings:
                            # Compute cosine similarity
                            similarity = float(np.dot(target_emb, candidate_emb))
                            
                            if similarity >= industry_similarity_threshold:
                                industry_candidates.add(ticker)
                                break  # Found a match, no need to check other target industries
                
                print(f"  Path A: Found {len(industry_candidates)} candidates by semantic industry matching (threshold={industry_similarity_threshold}): {target_similar_industries}")
        
        except Exception as e:
            print(f"  Warning: Semantic industry matching failed: {e}, falling back to substring matching")
            # Fallback to substring matching
            for similar_ind in target_similar_industries:
                matches = universe_df[
                    universe_df['industry'].str.contains(similar_ind, case=False, na=False) |
                    universe_df['sector'].str.contains(similar_ind, case=False, na=False)
                ]
                industry_candidates.update(matches['ticker'].tolist())
            print(f"  Path A: Found {len(industry_candidates)} candidates by similar_industries (substring fallback): {target_similar_industries}")
    # Fallback: Extract key industry terms from primary_industry_classification
    elif target_own_industry:
        # Extract key industry terms (e.g., "Research and Consulting Services" -> consulting, services)
        industry_keywords = []
        if 'consulting' in target_own_industry:
            industry_keywords.append('consulting')
        if 'services' in target_own_industry:
            industry_keywords.append('services')
        if 'research' in target_own_industry:
            industry_keywords.append('research')
        
        # Filter universe by own industry type
        if industry_keywords:
            for keyword in industry_keywords:
                matches = universe_df[
                    universe_df['industry'].str.contains(keyword, case=False, na=False) |
                    universe_df['sector'].str.contains(keyword, case=False, na=False)
                ]
                industry_candidates.update(matches['ticker'].tolist())
            print(f"  Path A: Found {len(industry_candidates)} candidates by own industry ({target.get('primary_industry_classification', '')})")
    
    # Note: Taxonomy-based fallback removed - using direct industry matching only
    # If no industry candidates found, Path A returns empty (other paths will still find candidates)
    
    # Note: We do NOT filter by customer industries (target.get('customer_industries', []))
    # Customer industries are for scoring similarity later (Feature I), not for initial filtering
    # Filtering by customer industries would give wrong results (e.g., Healthcare companies instead of consulting firms serving healthcare)
    
    # Path B: Semantic keyword search (NLP embeddings instead of substring matching)
    
    # Build target keyword text for embedding
    keyword_parts = []
    
    # Add products
    if target_products:
        keyword_parts.extend(target_products)
    
    # Add customers
    if target_customers:
        keyword_parts.extend(target_customers)
    
    # Add product_mix, business_activity, customer_segment if available
    product_mix = target.get('product_mix', {})
    if product_mix:
        for term, weight in product_mix.items():
            repeats = max(1, int(weight * 10))  # Weight by repetition
            keyword_parts.extend([term] * repeats)
    
    business_activity = target.get('business_activity', [])
    if business_activity:
        keyword_parts.extend(business_activity)
    
    customer_segment = target.get('customer_segment', target.get('customers', []))
    if customer_segment:
        keyword_parts.extend(customer_segment)
    
    target_keyword_text = ' '.join(keyword_parts).lower().strip()
    
    # Use NLP embeddings if FAISS is available
    top_keywords = []
    keyword_scores_list = []
    
    if index is not None and target_keyword_text:
        try:
            # Embed target keywords
            keyword_emb = get_cached_embedding(target_keyword_text, run_with_openai=run_with_openai, model='text-embedding-3-large')
            keyword_emb_np = np.array([keyword_emb]).astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(keyword_emb_np)
            
            # Search FAISS index (universe summaries already embedded)
            k = min(K_keywords * 2, index.ntotal)  # Get 2x for better coverage
            D, I = index.search(keyword_emb_np, k)
            
            # Build keyword scores from FAISS results
            for i, idx in enumerate(I[0]):
                if idx < len(meta_df):
                    row = meta_df.iloc[idx]
                    ticker = row['ticker']
                    similarity = float(D[0][i])
                    similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
                    
                    if similarity > 0:
                        # For P_kw and C_kw, use same similarity (can be refined later)
                        keyword_scores_list.append({
                            'ticker': ticker,
                            'P_kw': similarity,  # Semantic product similarity
                            'C_kw': similarity,  # Semantic customer similarity (same for now)
                            'keyword_score': similarity,
                            'path': 'B'
                        })
            
            keyword_df = pd.DataFrame(keyword_scores_list)
            if not keyword_df.empty:
                keyword_df = keyword_df.sort_values('keyword_score', ascending=False)
                top_keywords = keyword_df.head(K_keywords)['ticker'].tolist()
                print(f"  Path B: Found {len(top_keywords)} semantic keyword matches (NLP embeddings)")
            else:
                # No matches found, create empty keyword_df
                keyword_df = pd.DataFrame(columns=['ticker', 'P_kw', 'C_kw', 'keyword_score', 'path'])
                print(f"  Path B: No semantic keyword matches found (NLP embeddings)")
            
        except Exception as e:
            print(f"  Warning: Semantic keyword search failed: {e}, falling back to substring matching")
            # Fallback to substring matching
            keyword_scores = []
            for idx, row in universe_df.iterrows():
                summary = str(row.get('summary', '')).lower()
                P_kw = compute_keyword_overlap(summary, target_products)  # Taxonomy removed
                C_kw = compute_keyword_overlap(summary, target_customers)  # Taxonomy removed
                score = 0.45 * P_kw + 0.45 * C_kw
                keyword_scores.append({
                    'ticker': row['ticker'],
                    'P_kw': P_kw,
                    'C_kw': C_kw,
                    'keyword_score': score,
                    'path': 'B'
                })
            
            keyword_df = pd.DataFrame(keyword_scores)
            keyword_df = keyword_df.sort_values('keyword_score', ascending=False)
            top_keywords = keyword_df.head(K_keywords)['ticker'].tolist()
            print(f"  Path B: Found {len(top_keywords)} keyword matches (fallback to substring)")
    else:
        # Fallback to substring matching if FAISS not available
        keyword_scores = []
        for idx, row in universe_df.iterrows():
            summary = str(row.get('summary', '')).lower()
            P_kw = compute_keyword_overlap(summary, target_products)  # Taxonomy removed
            C_kw = compute_keyword_overlap(summary, target_customers)  # Taxonomy removed
            score = 0.45 * P_kw + 0.45 * C_kw
            keyword_scores.append({
                'ticker': row['ticker'],
                'P_kw': P_kw,
                'C_kw': C_kw,
                'keyword_score': score,
                'path': 'B'
            })
        
        keyword_df = pd.DataFrame(keyword_scores)
        keyword_df = keyword_df.sort_values('keyword_score', ascending=False)
        top_keywords = keyword_df.head(K_keywords)['ticker'].tolist()
        print(f"  Path B: Found {len(top_keywords)} keyword matches (fallback to substring - FAISS not available)")
    
    # Path C: Semantic sweep (ANN/FAISS)
    semantic_candidates = []
    if index is not None:
        target_text = target.get('text_profile', '')
        if target_text:
            try:
                query_vec = get_cached_embedding(target_text, run_with_openai=run_with_openai, model='text-embedding-3-large')
                query_vec = query_vec.reshape(1, -1)
                K_semantic_actual = min(K_semantic, index.ntotal)
                D, I = index.search(query_vec, K_semantic_actual)
                
                for idx, score in zip(I[0], D[0]):
                    if idx < len(meta_df):
                        row = meta_df.iloc[idx]
                        semantic_candidates.append({
                            'ticker': row['ticker'],
                            'S_fast': float(score),
                            'path': 'C'
                        })
            except Exception as e:
                print(f"Warning: Semantic search failed: {e}")
    
    semantic_df = pd.DataFrame(semantic_candidates)
    
    # Path D: Segment alias lift
    alias_candidates = []
    if not segments_alias.empty:
        for _, alias_row in segments_alias.iterrows():
            alias = alias_row.get('alias', '')
            for idx, row in universe_df.iterrows():
                summary = str(row.get('summary', '')).lower()
                if alias.lower() in summary:
                    alias_candidates.append({
                        'ticker': row['ticker'],
                        'path': 'D',
                        'alias': alias
                    })
    
    alias_df = pd.DataFrame(alias_candidates)
    
    # Union all paths
    all_tickers = set()
    all_tickers.update(industry_candidates)  # Path A: Cross-sector inclusion
    all_tickers.update(top_keywords)  # Path B: Keywords
    if not semantic_df.empty:
        all_tickers.update(semantic_df['ticker'].tolist())  # Path C: Semantic
    if not alias_df.empty:
        all_tickers.update(alias_df['ticker'].tolist())  # Path D: Alias
    
    # Epsilon exploration: add random candidates
    remaining = set(universe_df['ticker']) - all_tickers
    if remaining and epsilon_random > 0:
        np.random.seed(config.get('seed', 42))
        random_tickers = np.random.choice(list(remaining), min(epsilon_random, len(remaining)), replace=False)
        all_tickers.update(random_tickers)
        print(f"  Path E: Added {len(random_tickers)} random candidates (epsilon exploration)")
    
    print(f"  Union: {len(all_tickers)} total candidates from all paths")
    
    # Build segment vocabulary for semantic rank_key (if using semantic rank)
    # Try to load from cache first, then build if needed
    segment_vocabulary = None
    if use_semantic_rank and SEMANTIC_RANK_AVAILABLE:
        try:
            # Try to load cached segment vocabulary
            from features.segment_vocab_cache import (
                load_cached_vocabulary,
                save_cached_vocabulary,
                _get_company_fingerprint
            )
            
            # Build fingerprint for target
            target_fingerprint = _get_company_fingerprint(target)
            
            # For caching, we use a simplified approach: cache by target fingerprint
            # (since vocabulary is built from target + all candidates, we cache per target)
            cache_key = f"rank_key_{target_fingerprint}"
            cached_vocab = load_cached_vocabulary(cache_key, [target])
            
            if cached_vocab:
                segment_vocabulary = cached_vocab
                print(f"  Using cached segment vocabulary: {len(segment_vocabulary)} segments")
            else:
                # Build vocabulary from target + all candidates
                all_companies_for_vocab = [target]
                for ticker in all_tickers:
                    row = universe_df[universe_df['ticker'] == ticker].iloc[0]
                    all_companies_for_vocab.append({
                        'segment_mix': {},
                        'customer_segment': [row.get('sector', ''), row.get('industry', '')] if row.get('sector') or row.get('industry') else [],
                        'summary': row.get('summary', '')
                    })
                segment_vocabulary = build_segment_vocabulary(all_companies_for_vocab)
                
                # Save to cache
                save_cached_vocabulary(cache_key, [target], segment_vocabulary)
                print(f"  Built and cached segment vocabulary: {len(segment_vocabulary)} segments")
        except Exception as e:
            print(f"  Warning: Failed to load/build segment vocabulary: {e}, using fallback")
            segment_vocabulary = None
    
    # Build candidate DataFrame with all paths
    candidate_records = []
    failed_tickers = []
    for ticker in all_tickers:
        try:
            # Find row in universe - handle case where ticker might not exist
            ticker_matches = universe_df[universe_df['ticker'] == ticker]
            if ticker_matches.empty:
                failed_tickers.append({'ticker': ticker, 'reason': 'not in universe_df'})
                continue
            row = ticker_matches.iloc[0]
            paths = []
            
            # Check which paths contributed
            if ticker in industry_candidates:
                paths.append('A')  # Cross-sector inclusion
            if ticker in top_keywords:
                paths.append('B')  # Keywords
                # Try to get from keyword_df first (if it exists and has data)
                if 'keyword_df' in locals() and not keyword_df.empty and ticker in keyword_df['ticker'].values:
                    kw_row = keyword_df[keyword_df['ticker'] == ticker].iloc[0]
                    P_kw = kw_row['P_kw']
                    C_kw = kw_row['C_kw']
                    keyword_score = kw_row['keyword_score']
                elif keyword_scores_list and any(kw['ticker'] == ticker for kw in keyword_scores_list):
                    # Use from keyword_scores_list (NLP results)
                    kw_row = next(kw for kw in keyword_scores_list if kw['ticker'] == ticker)
                    P_kw = kw_row['P_kw']
                    C_kw = kw_row['C_kw']
                    keyword_score = kw_row['keyword_score']
                else:
                    # Fallback: compute on-the-fly
                    P_kw = compute_keyword_overlap(str(row.get('summary', '')).lower(), target_products)  # Taxonomy removed
                    C_kw = compute_keyword_overlap(str(row.get('summary', '')).lower(), target_customers)  # Taxonomy removed
                    keyword_score = 0.45 * P_kw + 0.45 * C_kw
            else:
                # Compute on-the-fly if not in keyword top-K
                P_kw = compute_keyword_overlap(str(row.get('summary', '')).lower(), target_products)  # Taxonomy removed
                C_kw = compute_keyword_overlap(str(row.get('summary', '')).lower(), target_customers)  # Taxonomy removed
                keyword_score = 0.45 * P_kw + 0.45 * C_kw
            
            if not semantic_df.empty and ticker in semantic_df['ticker'].values:
                paths.append('C')  # Semantic
                sem_row = semantic_df[semantic_df['ticker'] == ticker].iloc[0]
                S_fast = sem_row['S_fast']
            else:
                S_fast = 0.0
            
            if not alias_df.empty and ticker in alias_df['ticker'].values:
                paths.append('D')  # Alias
            
            # Compute rank_key for shortlisting
            # NEW: Use sophisticated semantic rank_key if available, otherwise fallback to old keyword-based
            if use_semantic_rank and SEMANTIC_RANK_AVAILABLE:
                try:
                    # Build candidate data dict for semantic rank_key
                    candidate_data_dict = {
                        'ticker': ticker,
                        'summary': row.get('summary', ''),
                        'industry': row.get('industry', ''),
                        'sector': row.get('sector', ''),
                        'customer_segment': [row.get('sector', ''), row.get('industry', '')] if row.get('sector') or row.get('industry') else [],
                        'segment_mix': {}  # Will be inferred from customer_segment if available
                    }
                    
                    # Compute semantic rank_key
                    rank_result = compute_rank_key_semantic(
                        target,
                        candidate_data_dict,
                        config=config,
                        vocabulary=segment_vocabulary,
                        run_with_openai=run_with_openai
                    )
                    
                    rank_key = rank_result['rank_key']
                    P_fast = rank_result['P_fast']
                    C_mix = rank_result['C_mix']
                    M_fast = rank_result['M_fast']
                    entropy_penalty = rank_result['entropy_penalty']
                    entropy_target = rank_result['entropy_target']
                    entropy_candidate = rank_result['entropy_candidate']
                    industry_bonus = rank_result.get('industry_bonus', 0.0)  # Industry match bonus
                except Exception as e:
                    # Fallback to old keyword-based rank_key on error
                    print(f"  Warning: Semantic rank_key failed for {ticker}: {e}, using fallback")
                    rank_key = 0.45 * P_kw + 0.45 * C_kw + 0.10 * S_fast
                    P_fast = P_kw  # Approximate
                    C_mix = C_kw  # Approximate
                    M_fast = S_fast  # Approximate
                    entropy_penalty = 0.0
                    entropy_target = 0.0
                    entropy_candidate = 0.0
                    industry_bonus = 0.0  # Fallback: no industry bonus
            else:
                # Old keyword-based rank_key (fallback)
                rank_key = 0.45 * P_kw + 0.45 * C_kw + 0.10 * S_fast
                P_fast = P_kw  # Approximate
                C_mix = C_kw  # Approximate
                M_fast = S_fast  # Approximate
                entropy_penalty = 0.0
                entropy_target = 0.0
                entropy_candidate = 0.0
                industry_bonus = 0.0  # Fallback: no industry bonus
            
            candidate_records.append({
                'ticker': ticker,
            'name': row.get('name', ''),
            'exchange': row.get('exchange', ''),
            'sector': row.get('sector', ''),
            'industry': row.get('industry', ''),
            'summary': row.get('summary', ''),
            'P_kw': P_kw,  # Keep for backward compatibility / logging
            'C_kw': C_kw,  # Keep for backward compatibility / logging
            'S_fast': S_fast,  # Keep for backward compatibility / logging
            'keyword_score': keyword_score,  # Keep for backward compatibility / logging
            'rank_key': rank_key,  # NEW: Semantic rank_key
            'P_fast': P_fast,  # NEW: Product similarity (semantic)
            'C_mix': C_mix,  # NEW: Segment mix similarity
            'M_fast': M_fast,  # NEW: Model similarity
            'entropy_penalty': entropy_penalty,  # NEW: Entropy mismatch penalty
            'entropy_target': entropy_target,  # NEW: Target entropy
            'entropy_candidate': entropy_candidate,  # NEW: Candidate entropy
            'industry_bonus': industry_bonus,  # NEW: Industry match bonus (prioritized)
            'paths': ','.join(paths) if paths else 'A',
            'cik': row.get('cik', ''),
            'website': row.get('website', ''),
            'permitted_industry_match': ticker in industry_candidates,  # Flag for Path A
            # Include Path B semantic evidence if available (from keyword_df)
            'path_b_summary': keyword_df[keyword_df['ticker'] == ticker]['path_b_summary'].iloc[0] if (ticker in top_keywords and 'keyword_df' in locals() and not keyword_df.empty and ticker in keyword_df['ticker'].values and 'path_b_summary' in keyword_df.columns) else '',
            'path_b_similarity': keyword_df[keyword_df['ticker'] == ticker]['path_b_similarity'].iloc[0] if (ticker in top_keywords and 'keyword_df' in locals() and not keyword_df.empty and ticker in keyword_df['ticker'].values and 'path_b_similarity' in keyword_df.columns) else None,
                'path_b_target_keywords': keyword_df[keyword_df['ticker'] == ticker]['path_b_target_keywords'].iloc[0] if (ticker in top_keywords and 'keyword_df' in locals() and not keyword_df.empty and ticker in keyword_df['ticker'].values and 'path_b_target_keywords' in keyword_df.columns) else ''
            })
        except Exception as e:
            # Log failed tickers but continue processing others
            failed_tickers.append({'ticker': ticker, 'reason': str(e)})
            continue
    
    if failed_tickers:
        print(f"  ⚠️  Failed to process {len(failed_tickers)} tickers: {[f['ticker'] for f in failed_tickers[:10]]}")
        if len(failed_tickers) > 10:
            print(f"      ... and {len(failed_tickers) - 10} more")
    
    candidates_df = pd.DataFrame(candidate_records)
    candidates_df = candidates_df.sort_values('rank_key', ascending=False)
    
    # Enforce diversity: keep at least 20-30% outside top industry
    # BUT: Always preserve top-ranked companies regardless of industry
    if len(candidates_df) > 0:
        top_industry = candidates_df['industry'].mode()[0] if not candidates_df['industry'].mode().empty else ''
        diversity_cut = max(int(len(candidates_df) * 0.2), 10)
        
        # Always keep top 80 by rank_key (shortlist size) regardless of industry
        top_80_by_rank = candidates_df.head(80)  # Always keep top 80
        
        # For the remaining slots (80 to K_total), enforce diversity
        remaining_slots = max(0, K_total - 80)
        if remaining_slots > 0:
            # Get companies ranked 81+ that are NOT in top industry
            remaining_candidates = candidates_df.iloc[80:]
            diverse = remaining_candidates[~remaining_candidates['industry'].isin([top_industry])].head(min(diversity_cut, remaining_slots))
            
            # Also include companies ranked 81-100 even if they're in top industry
            # This ensures companies just outside top 80 aren't lost
            near_top = remaining_candidates.head(20)  # Include next 20 regardless of industry
            
            # Combine: top 80 + near-top (81-100) + diverse companies
            candidates_df = pd.concat([top_80_by_rank, near_top, diverse]).drop_duplicates(subset=['ticker'])
        else:
            # If K_total <= 80, just take top K_total
            candidates_df = candidates_df.head(K_total)
        
        # Final cap at K_total
        candidates_df = candidates_df.head(K_total)
    
    return candidates_df


if __name__ == "__main__":
    # Test
    import sys
    import json
    target = {
        "products": ["Payment Processing", "Transaction APIs"],
        "customers": ["Banks", "Retailers"],
        "text_profile": "Target is a fintech offering payment APIs to banks and retailers."
    }
    config = {
        "recall": {
            "K_total": 50,
            "K_semantic": 20,
            "K_keywords": 30,
            "K_alias": 10,
            "epsilon_random": 5
        },
        "seed": 42
    }
    candidates = generate_candidates(target, config)
    print(f"Generated {len(candidates)} candidates")
    print(candidates[['ticker', 'name', 'rank_key', 'paths']].head(10))
