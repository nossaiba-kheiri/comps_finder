"""
prelim_filter.py: Fast Preliminary Filter for candidate selection.

Given a target company and universe, selects ~200-300 preliminary candidates using:
- Path A: Semantic KNN (FAISS vector search)
- Path B: Keyword overlap (product-mix aware)
- Path C: Sector/Industry/Country signals
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import faiss
from typing import Dict, List, Set, Optional

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from universe.embeddings_index import get_cached_embedding, preprocess

# Paths
UNIVERSE_PATH = os.path.join(ROOT, 'data/universe_us.csv')
FAISS_PATH = os.path.join(ROOT, 'data/embeddings/universe_index.faiss')
META_PATH = os.path.join(ROOT, 'data/embeddings/universe_meta.parquet')


def basic_clean(text: str) -> str:
    """Basic text cleaning: lowercase, strip HTML, remove punctuation, normalize whitespace."""
    if not text:
        return ""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def construct_target_profile_text(target: Dict) -> str:
    """
    Construct target_profile_text from product_mix, business_activity, and customer_segment.
    
    Returns cleaned text ready for embedding.
    """
    parts = []
    
    # (a) Product mix sentence
    product_mix = target.get('product_mix', {})
    if product_mix:
        items = sorted(product_mix.items(), key=lambda x: x[1], reverse=True)
        mix_parts = []
        for term, weight in items:
            pct = int(weight * 100)
            mix_parts.append(f"{pct}% in {term}")
        if mix_parts:
            parts.append(f"Approximately {', '.join(mix_parts[:-1])}, and {mix_parts[-1]}. ")
    
    # (b) Products/services sentence
    business_activity = target.get('business_activity', [])
    if business_activity:
        if len(business_activity) == 1:
            products_sentence = f"We provide {business_activity[0]}. "
        else:
            products_list = ', '.join(business_activity[:-1]) + f", and {business_activity[-1]}"
            products_sentence = f"We provide {products_list} services. "
        parts.append(products_sentence)
    
    # (c) Customer segments sentence
    customer_segment = target.get('customer_segment', [])
    if customer_segment:
        if len(customer_segment) == 1:
            customers_sentence = f"We serve {customer_segment[0]}. "
        else:
            customers_list = ', '.join(customer_segment[:-1]) + f", and {customer_segment[-1]}"
            customers_sentence = f"We serve {customers_list}. "
        parts.append(customers_sentence)
    
    # Use existing text_profile if present, otherwise use constructed
    if target.get('text_profile'):
        # Option: append or replace (for now, append)
        parts.append(target.get('text_profile'))
    
    target_profile_text = ' '.join(parts)
    return basic_clean(target_profile_text)


def build_target_keywords(target: Dict) -> List[Dict]:
    """
    Build target_keywords list with weights from product_mix, business_activity, customer_segment.
    
    Returns list of {"term": str, "weight": float}
    """
    keywords = []
    
    # Product mix terms (higher weights)
    product_mix = target.get('product_mix', {})
    for term, weight in product_mix.items():
        keywords.append({"term": term.lower(), "weight": float(weight)})
    
    # Business activities (medium weight)
    business_activity = target.get('business_activity', [])
    for term in business_activity:
        keywords.append({"term": term.lower(), "weight": 0.3})
    
    # Customer segments (lower weight)
    customer_segment = target.get('customer_segment', [])
    for term in customer_segment:
        keywords.append({"term": term.lower(), "weight": 0.2})
    
    return keywords


def compute_kw_score(summary_clean: str, target_keywords: List[Dict]) -> float:
    """
    Compute keyword score for a company summary.
    
    Returns raw score (not normalized).
    """
    if not summary_clean or not target_keywords:
        return 0.0
    
    score = 0.0
    summary_lower = summary_clean.lower()
    
    for kw in target_keywords:
        term = kw['term']
        weight = kw['weight']
        
        # Check if term appears (substring or word-level match)
        if term in summary_lower:
            # Word boundary check for better matching
            import re
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, summary_lower):
                score += weight
    
    return score


def sector_contains_hint(sector: str, hint: str) -> bool:
    """Check if sector contains hint (fuzzy matching)."""
    if not sector or not hint:
        return False
    sector_lower = sector.lower()
    hint_lower = hint.lower()
    return hint_lower in sector_lower or sector_lower in hint_lower


def important_tokens_from(product_mix_keys: List[str], business_activity: List[str]) -> List[str]:
    """Extract important tokens from product_mix and business_activity."""
    tokens = []
    for term in product_mix_keys + business_activity:
        # Split by common separators and add individual words
        words = term.lower().replace('-', ' ').replace('_', ' ').split()
        tokens.extend([w for w in words if len(w) >= 3])
    return list(set(tokens))


def prelim_filter(target: Dict, config: Dict, run_with_openai: bool = False) -> pd.DataFrame:
    """
    Fast preliminary filter: select ~200-300 candidates from universe.
    
    Args:
        target: Target company dict with product_mix, business_activity, customer_segment, etc.
        config: Config dict with prelim_filter section
        run_with_openai: Whether to use real OpenAI embeddings
    
    Returns:
        DataFrame with preliminary candidates, sorted by score_pre desc
    """
    # Load config
    prelim_config = config.get('prelim_filter', {})
    K_semantic = prelim_config.get('K_semantic', 300)
    K_keyword = prelim_config.get('K_keyword', 200)
    N_prelim = prelim_config.get('N_prelim', 250)
    w_semantic = prelim_config.get('w_semantic', 0.5)
    w_keyword = prelim_config.get('w_keyword', 0.25)
    w_sector = prelim_config.get('w_sector', 0.15)
    w_industry = prelim_config.get('w_industry', 0.10)
    w_country = prelim_config.get('w_country', 0.0)
    
    # Load universe
    print(f"  Loading universe...")
    universe_df = pd.read_csv(UNIVERSE_PATH)
    universe_df = universe_df.fillna('')
    print(f"  Loaded {len(universe_df)} companies")
    
    # Clean summaries (if not already done)
    if 'summary_clean' not in universe_df.columns:
        print(f"  Cleaning summaries...")
        universe_df['summary_clean'] = universe_df['summary'].apply(basic_clean)
    
    # 1. Construct target profile text and embed
    print(f"  Constructing target profile...")
    target_profile_text = construct_target_profile_text(target)
    target_emb = get_cached_embedding(target_profile_text, run_with_openai=run_with_openai)
    
    # 2. Path A - Semantic KNN
    print(f"  Path A: Semantic KNN (top {K_semantic})...")
    S_fast = {}
    
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        try:
            index = faiss.read_index(FAISS_PATH)
            meta_df = pd.read_parquet(META_PATH)
            
            # Query FAISS
            target_emb_np = np.array([target_emb]).astype(np.float32)
            # Normalize for cosine similarity (FAISS IndexFlatIP expects normalized vectors)
            faiss.normalize_L2(target_emb_np)
            
            k = min(K_semantic, len(meta_df))
            distances, indices = index.search(target_emb_np, k)
            
            # Convert distances to similarities (for IndexFlatIP, higher is better)
            for i, idx in enumerate(indices[0]):
                if idx < len(meta_df):
                    ticker = meta_df.iloc[idx]['ticker']
                    # For IndexFlatIP, distance is already similarity (higher = more similar)
                    similarity = float(distances[0][i])
                    S_fast[ticker] = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
            print(f"    Found {len(S_fast)} semantic neighbors")
        except Exception as e:
            print(f"    Warning: FAISS search failed: {e}")
            S_fast = {}
    else:
        print(f"    Warning: FAISS index not found, skipping semantic search")
        S_fast = {}
    
    semantic_hits = set(S_fast.keys())
    
    # 3. Path B - Semantic keyword search (NLP embeddings)
    print(f"  Path B: Semantic keyword search (NLP embeddings, top {K_keyword})...")
    
    # Build target keyword text from product_mix, business_activity, customer_segment
    keyword_parts = []
    
    # Add product mix terms (weighted)
    product_mix = target.get('product_mix', {})
    if product_mix:
        for term, weight in product_mix.items():
            # Repeat term based on weight for emphasis
            repeats = max(1, int(weight * 10))
            keyword_parts.extend([term] * repeats)
    
    # Add business activities
    business_activity = target.get('business_activity', [])
    keyword_parts.extend(business_activity)
    
    # Add customer segments
    customer_segment = target.get('customer_segment', [])
    keyword_parts.extend(customer_segment)
    
    # Also add products if available
    products = target.get('products', [])
    if products:
        keyword_parts.extend(products)
    
    # Construct keyword text for embedding
    target_keyword_text = ' '.join(keyword_parts).lower().strip()
    
    KW_score = {}
    
    # Use FAISS for semantic keyword matching if available
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH) and target_keyword_text:
        try:
            # Embed target keywords
            keyword_emb = get_cached_embedding(target_keyword_text, run_with_openai=run_with_openai)
            
            # Query FAISS index (same as Path A, but with keyword text instead of full profile)
            index = faiss.read_index(FAISS_PATH)
            meta_df = pd.read_parquet(META_PATH)
            
            keyword_emb_np = np.array([keyword_emb]).astype(np.float32)
            faiss.normalize_L2(keyword_emb_np)  # Normalize for cosine similarity
            
            # Search top K_keyword results (might be more than needed, but that's okay)
            k = min(K_keyword * 2, len(meta_df))  # Get 2x for better coverage
            distances, indices = index.search(keyword_emb_np, k)
            
            # Convert to similarity scores and store evidence
            path_b_evidence = {}  # Store semantic evidence for explainability
            for i, idx in enumerate(indices[0]):
                if idx < len(meta_df):
                    row = meta_df.iloc[idx]
                    ticker = row['ticker']
                    similarity = float(distances[0][i])
                    # Clamp to [0, 1] and only keep positive similarities
                    similarity = max(0.0, min(1.0, similarity))
                    if similarity > 0:
                        KW_score[ticker] = similarity
                        # Store semantic evidence: summary text, similarity, target keywords
                        summary_text = row.get('summary', '')[:500]  # Truncate for storage
                        path_b_evidence[ticker] = {
                            'similarity': similarity,
                            'summary_text': summary_text,
                            'target_keywords': target_keyword_text,
                            'path': 'B',
                            'method': 'nlp_embedding'
                        }
            
            # Normalize scores
            if KW_score:
                max_kw = max(KW_score.values())
                KW_score = {ticker: score / max_kw for ticker, score in KW_score.items()}
            
            print(f"    Found {len(KW_score)} semantic keyword matches (using NLP embeddings)")
            
        except Exception as e:
            print(f"    Warning: Semantic keyword search failed: {e}, falling back to substring matching")
            # Fallback to substring matching
            target_keywords = build_target_keywords(target)
            KW_raw = {}
            for _, row in universe_df.iterrows():
                ticker = row['ticker']
                summary_clean = row.get('summary_clean', '')
                score = compute_kw_score(summary_clean, target_keywords)
                if score > 0:
                    KW_raw[ticker] = score
            max_kw = max(KW_raw.values()) if KW_raw else 1.0
            KW_score = {ticker: score / max_kw for ticker, score in KW_raw.items()}
            print(f"    Found {len(KW_score)} keyword matches (fallback to substring)")
    else:
        # Fallback to substring matching if FAISS not available
        target_keywords = build_target_keywords(target)
        KW_raw = {}
        for _, row in universe_df.iterrows():
            ticker = row['ticker']
            summary_clean = row.get('summary_clean', '')
            score = compute_kw_score(summary_clean, target_keywords)
            if score > 0:
                KW_raw[ticker] = score
        max_kw = max(KW_raw.values()) if KW_raw else 1.0
        KW_score = {ticker: score / max_kw for ticker, score in KW_raw.items()}
        print(f"    Found {len(KW_score)} keyword matches (fallback to substring - FAISS not available)")
    
    # Get top K_keyword by KW_score
    sorted_kw = sorted(KW_score.items(), key=lambda x: x[1], reverse=True)
    keyword_hits = set([ticker for ticker, _ in sorted_kw[:K_keyword]])
    print(f"    Selected top {len(keyword_hits)} keyword matches")
    
    # Store Path B evidence for later use (if NLP was used)
    if 'path_b_evidence' in locals() and path_b_evidence:
        # Make path_b_evidence accessible to caller
        prelim_filter.path_b_evidence = path_b_evidence
    else:
        prelim_filter.path_b_evidence = {}
    
    # 4. Path C - Sector/Industry/Country signals
    print(f"  Path C: Sector/Industry/Country matching...")
    
    # CRITICAL: Filter by OWN industry first, not by customer industries
    # Customer industries are for scoring (Feature I), not for filtering
    target_own_industry = target.get('primary_industry_classification', '').lower()
    target_similar_industries = target.get('similar_industries', [])  # Similar own industries for filtering
    
    # Pre-filter universe by own industry if possible
    # This ensures we compare companies in the same industry (e.g., consulting firms)
    filtered_universe_df = universe_df.copy()
    
    if target_similar_industries:
        # Use similar_industries for filtering (more accurate than extracting from primary_industry_classification)
        print(f"    Target similar industries: {target_similar_industries}")
        print(f"    Using similar_industries for filtering candidates by own industry")
    elif target_own_industry:
        # Fallback: Extract key terms from own industry (e.g., "Research and Consulting Services" -> ["consulting", "services"])
        industry_terms = []
        if 'consulting' in target_own_industry:
            industry_terms.append('consulting')
        if 'services' in target_own_industry:
            industry_terms.append('services')
        if 'research' in target_own_industry:
            industry_terms.append('research')
        
        # If we can identify industry terms, filter universe
        # But be lenient - don't filter too strictly (keep semantic/keyword candidates too)
        if industry_terms:
            print(f"    Target own industry: {target.get('primary_industry_classification', '')}")
            print(f"    Filtering candidates by own industry terms: {industry_terms}")
            # Note: We still use full universe for semantic/keyword paths, but boost industry matches
    
    sector_match = {}
    industry_match = {}
    country_match = {}
    
    target_sector_hint = target.get('sector_hint', '')
    target_country = target.get('country', '')
    
    # For industry matching: match by own industry, not customer industries
    sector_hits = set()
    industry_hits = set()
    
    for _, row in universe_df.iterrows():
        ticker = row['ticker']
        sector = str(row.get('sector', '')).lower()
        industry = str(row.get('industry', '')).lower()
        country = str(row.get('country', ''))
        
        # Sector match
        if target_sector_hint:
            sector_match[ticker] = 1 if sector_contains_hint(sector, target_sector_hint) else 0
            if sector_match.get(ticker, 0) == 1:
                sector_hits.add(ticker)
        else:
            sector_match[ticker] = 0
        
        # Industry match: Match by OWN industry (similar_industries), not customer industries
        # This ensures we boost candidates in the same industry as target
        industry_match[ticker] = 0
        if target_similar_industries:
            # Use similar_industries for matching (more accurate)
            for similar_ind in target_similar_industries:
                similar_ind_lower = similar_ind.lower()
                # Check if candidate's industry contains similar industry term
                if similar_ind_lower in industry or industry in similar_ind_lower:
                    industry_match[ticker] = 1
                    industry_hits.add(ticker)
                    break
        elif target_own_industry and industry:
            # Fallback: Check if candidate's industry matches target's own industry
            # Use fuzzy matching (contains, word overlap)
            industry_terms = [term.strip() for term in target_own_industry.split() if len(term.strip()) > 3]
            for term in industry_terms:
                if term in industry:
                    industry_match[ticker] = 1
                    industry_hits.add(ticker)
                    break
            
            # Also check for common industry type keywords (consulting, services, etc.)
            if 'consulting' in target_own_industry and 'consulting' in industry:
                industry_match[ticker] = 1
                industry_hits.add(ticker)
            elif 'services' in target_own_industry and 'services' in industry:
                industry_match[ticker] = 1
                industry_hits.add(ticker)
        
        # Country match
        if target_country:
            country_match[ticker] = 1 if country == target_country else 0
        else:
            country_match[ticker] = 0
    
    sector_industry_hits = sector_hits | industry_hits
    print(f"    Found {len(sector_hits)} sector matches, {len(industry_hits)} industry matches")
    
    # 5. Combine paths
    prelim_candidates = semantic_hits | keyword_hits | sector_industry_hits
    print(f"  Combined: {len(prelim_candidates)} preliminary candidates")
    
    # 6. Compute normalized scores and score_pre
    print(f"  Computing preliminary scores...")
    
    # Normalize S_fast and KW_score on prelim_candidates only
    if prelim_candidates:
        s_values = [S_fast.get(t, 0.0) for t in prelim_candidates]
        kw_values = [KW_score.get(t, 0.0) for t in prelim_candidates]
        max_s = max(s_values) if s_values and max(s_values) > 0 else 1.0
        max_kw = max(kw_values) if kw_values and max(kw_values) > 0 else 1.0
    else:
        max_s = 1.0
        max_kw = 1.0
    
    # Build preliminary candidates DataFrame
    candidate_rows = []
    for ticker in prelim_candidates:
        row = universe_df[universe_df['ticker'] == ticker].iloc[0] if len(universe_df[universe_df['ticker'] == ticker]) > 0 else None
        if row is None:
            continue
        
        S_norm = (S_fast.get(ticker, 0.0) / max_s) if max_s > 0 else 0.0
        KW_norm = (KW_score.get(ticker, 0.0) / max_kw) if max_kw > 0 else 0.0
        
        # CRITICAL FIX: Penalize companies without industry match
        # Companies in wrong industry should not rank #1, even with high semantic score
        # This prevents e.g., Amazon (Internet Retail) from ranking above consulting firms
        industry_penalty = 0.0
        if industry_match.get(ticker, 0) == 0:
            # Company doesn't match industry - apply penalty
            # This is critical for filtering: we want companies in the SAME industry
            if ticker in semantic_hits:
                # Company came from semantic path but doesn't match industry
                # This is suspicious - likely wrong match (e.g., Amazon for consulting target)
                # Apply strong penalty: reduce semantic contribution by 75%
                industry_penalty = 0.75  # Strong penalty for wrong industry + semantic match
            else:
                # Company doesn't match industry and wasn't even a semantic match
                # Still apply penalty but less severe (might be from keyword path)
                # However, if we have industry matches available, prioritize them
                # Reduce all contributions by 30% if no industry match
                industry_penalty = 0.30  # Moderate penalty for wrong industry
        
        score_pre = (
            w_semantic * S_norm * (1.0 - industry_penalty) +
            w_keyword * KW_norm +
            w_sector * sector_match.get(ticker, 0) +
            w_industry * industry_match.get(ticker, 0) +
            w_country * country_match.get(ticker, 0)
        )
        
        candidate_rows.append({
            'ticker': ticker,
            'name': row.get('name', ''),
            'exchange': row.get('exchange', ''),
            'sector': row.get('sector', ''),
            'industry': row.get('industry', ''),
            'country': row.get('country', ''),
            'summary': row.get('summary', ''),
            'website': row.get('website', ''),
            'S_fast': S_fast.get(ticker, 0.0),
            'KW_score': KW_score.get(ticker, 0.0),
            'sector_match': sector_match.get(ticker, 0),
            'industry_match': industry_match.get(ticker, 0),
            'country_match': country_match.get(ticker, 0),
            'score_pre': score_pre,
            'path_b_evidence': json.dumps(prelim_filter.path_b_evidence.get(ticker, {}))  # Store Path B semantic evidence
        })
    
    prelim_df = pd.DataFrame(candidate_rows)
    
    # Sort by score_pre desc and keep top N_prelim
    if len(prelim_df) > 0:
        prelim_df = prelim_df.sort_values('score_pre', ascending=False)
        prelim_df = prelim_df.head(N_prelim)
        prelim_df = prelim_df.reset_index(drop=True)
    
    print(f"  ✓ Selected {len(prelim_df)} preliminary candidates (top {N_prelim})")
    
    # Logging
    print(f"\n  Path contributions:")
    print(f"    Semantic hits: {len(semantic_hits)}")
    print(f"    Keyword hits: {len(keyword_hits)}")
    print(f"    Sector/Industry hits: {len(sector_industry_hits)}")
    print(f"    Final union: {len(prelim_candidates)}")
    
    return prelim_df


if __name__ == "__main__":
    # Test
    import json
    import yaml
    
    target = {
        "name": "Huron Consulting Group",
        "url": "https://www.huronconsultinggroup.com",
        "product_mix": {
            "Healthcare consulting": 0.55,
            "Education analytics": 0.25,
            "Corporate digital transformation": 0.20
        },
        "business_activity": [
            "EHR implementation",
            "Revenue cycle optimization",
            "ERP consulting"
        ],
        "customer_segment": [
            "Hospitals",
            "Health systems",
            "Universities"
        ],
        "country": "US",
        "sector_hint": "Healthcare/Services",
        "text_profile": "Huron Consulting Group provides healthcare and education consulting services.",
        "mode": "all_segments"
    }
    
    config_path = os.path.join(ROOT, 'config/runtime.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    prelim_df = prelim_filter(target, config, run_with_openai=False)
    print(f"\nTop 10 preliminary candidates:")
    print(prelim_df[['ticker', 'name', 'score_pre', 'S_fast', 'KW_score']].head(10))

