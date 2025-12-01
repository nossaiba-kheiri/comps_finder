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
    
    # (a) Business model emphasis (generic - extract from similar_industries)
    business_model_type = target.get('business_model_type', '').lower()
    services_share = target.get('services_share_estimate', 0.5)
    similar_industries = target.get('similar_industries', [])
    
    # Extract industry-specific terms from similar_industries for better semantic matching
    # This works for any industry, not just consulting
    if similar_industries:
        # Extract key terms from industry names (e.g., "Consulting Services" -> "consulting", "services")
        industry_terms = []
        for industry in similar_industries:
            industry_lower = industry.lower()
            # Split by common separators and extract meaningful words
            words = industry_lower.replace('-', ' ').replace('_', ' ').split()
            # Keep words that are meaningful (length >= 4, not common stop words)
            meaningful_words = [w for w in words if len(w) >= 4 and w not in ['and', 'the', 'for', 'with']]
            industry_terms.extend(meaningful_words)
        
        if industry_terms:
            # Add unique industry terms to profile
            unique_terms = list(set(industry_terms))[:5]  # Limit to top 5 unique terms
            parts.append(' '.join(unique_terms) + '. ')
    
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
    
    # (d) Similar industries (for better matching)
    similar_industries = target.get('similar_industries', [])
    if similar_industries:
        industries_text = ', '.join(similar_industries)
        parts.append(f"Industry: {industries_text}. ")
    
    # Use existing text_profile if present, otherwise use constructed
    if target.get('text_profile'):
        # Option: append or replace (for now, append)
        parts.append(target.get('text_profile'))
    
    target_profile_text = ' '.join(parts)
    
    # Fallback: If no profile text was constructed, use business_description
    if not target_profile_text or len(target_profile_text.strip()) == 0:
        business_description = target.get('business_description', '')
        if business_description:
            target_profile_text = business_description
        else:
            # Last resort: use company name
            target_profile_text = target.get('name', '')
    
    return basic_clean(target_profile_text)


def build_target_keywords(target: Dict) -> List[Dict]:
    """
    Build target_keywords list with weights from product_mix, business_activity, customer_segment.
    
    Returns list of {"term": str, "weight": float}
    """
    keywords = []
    
    # Add industry-specific keywords extracted from similar_industries (generic approach)
    business_model_type = target.get('business_model_type', '').lower()
    services_share = target.get('services_share_estimate', 0.5)
    similar_industries = target.get('similar_industries', [])
    
    # Extract industry-specific terms from similar_industries
    # This works for any industry: consulting, software, manufacturing, etc.
    if similar_industries:
        industry_keywords = []
        for industry in similar_industries:
            industry_lower = industry.lower()
            # Extract meaningful terms from industry name
            words = industry_lower.replace('-', ' ').replace('_', ' ').split()
            meaningful_words = [w for w in words if len(w) >= 4 and w not in ['and', 'the', 'for', 'with', 'services']]
            industry_keywords.extend(meaningful_words)
            
            # Add common business suffixes/patterns that appear in company names
            # These are generic and work for any industry
            common_business_terms = ['group', 'partners', 'solutions', 'systems', 'services', 
                                    'technologies', 'holdings', 'corporation', 'company']
            # Only add if they appear in the industry name (to avoid adding irrelevant terms)
            for term in common_business_terms:
                if term in industry_lower:
                    industry_keywords.append(term)
        
        # Add unique industry keywords with high weight
        unique_industry_keywords = list(set(industry_keywords))
        for term in unique_industry_keywords:
            keywords.append({"term": term.lower(), "weight": 0.4})  # High weight for industry terms
    
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
    
    # Similar industries (for industry matching)
    similar_industries = target.get('similar_industries', [])
    for industry in similar_industries:
        # Extract key terms from industry names
        industry_lower = industry.lower()
        if 'consulting' in industry_lower:
            keywords.append({"term": "consulting", "weight": 0.5})
        if 'services' in industry_lower:
            keywords.append({"term": "services", "weight": 0.3})
        # Add full industry name as keyword
        keywords.append({"term": industry_lower, "weight": 0.3})
    
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
    
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH) and target_emb is not None:
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
    elif target_emb is None:
        print(f"    Warning: Target embedding failed (likely rate limit), skipping semantic search")
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
    
    # Use business_activity instead of products to avoid bias
    # Products (e.g., "enterprise health record") can match across different business models
    # Business activity (e.g., "consulting services") better captures what the company DOES
    # Note: business_activity was already added above, so we skip products here
    
    # Construct keyword text for embedding
    target_keyword_text = ' '.join(keyword_parts).lower().strip()
    
    KW_score = {}
    
    # Use FAISS for semantic keyword matching if available
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH) and target_keyword_text:
        try:
            # Embed target keywords
            keyword_emb = get_cached_embedding(target_keyword_text, run_with_openai=run_with_openai)
            
            # Skip if embedding failed (e.g., rate limit)
            if keyword_emb is None:
                raise ValueError("Embedding returned None (likely rate limit)")
            
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
    
    # Path B.5: Additional industry-specific company search (generic)
    # This helps find companies in the same industry that might not match on exact keywords
    business_model_type = target.get('business_model_type', '').lower()
    services_share = target.get('services_share_estimate', 0.5)
    similar_industries = target.get('similar_industries', [])
    
    # Only do this for services-based companies (consulting, professional services, etc.)
    # For software/manufacturing, the semantic/keyword search should be sufficient
    if (business_model_type in ['services', 'hybrid_services_software'] or services_share >= 0.5) and similar_industries:
        industry_company_hits = set()
        
        # Extract industry keywords from similar_industries (fully generic)
        industry_keywords = []
        for industry in similar_industries:
            industry_lower = industry.lower()
            # Extract meaningful words from industry name
            words = industry_lower.replace('-', ' ').replace('_', ' ').split()
            meaningful_words = [w for w in words if len(w) >= 4 and w not in ['and', 'the', 'for', 'with']]
            industry_keywords.extend(meaningful_words)
            
            # Add common business suffixes/patterns (generic, not industry-specific)
            # These help match company names (e.g., "Group", "Partners", "Solutions", "Systems")
            common_suffixes = ['group', 'partners', 'solutions', 'systems', 'services', 'technologies', 'holdings']
            # Only add if they're relevant to the industry (e.g., if industry has "services", add "services")
            for suffix in common_suffixes:
                if suffix in industry_lower:
                    industry_keywords.append(suffix)
        
        unique_industry_keywords = list(set(industry_keywords))
        
        for _, row in universe_df.iterrows():
            ticker = row['ticker']
            name = str(row.get('name', '')).lower()
            industry = str(row.get('industry', '')).lower()
            summary = str(row.get('summary', '')).lower()
            
            # Check if company name contains industry keywords
            name_has_industry = any(kw in name for kw in unique_industry_keywords)
            # Check if industry matches target's similar industries
            industry_matches = any(target_ind.lower() in industry or industry in target_ind.lower() 
                                  for target_ind in similar_industries)
            # Check if summary mentions industry terms prominently
            summary_has_industry = any(kw in summary for kw in unique_industry_keywords[:3])  # Top 3 keywords
            
            if (name_has_industry or industry_matches or summary_has_industry):
                industry_company_hits.add(ticker)
        
        # Add industry company hits to keyword_hits (up to additional 50)
        additional_industry = industry_company_hits - keyword_hits
        if additional_industry:
            # Add top scoring ones (if they have any KW_score) or just add them all
            additional_sorted = sorted(
                [(t, KW_score.get(t, 0.1)) for t in additional_industry],
                key=lambda x: x[1],
                reverse=True
            )[:50]  # Add up to 50 additional industry companies
            keyword_hits.update([t for t, _ in additional_sorted])
            print(f"    Found {len(additional_industry)} additional industry companies, added {len(additional_sorted)}")
    
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
        # Fallback: Extract key terms from own industry (generic - works for any industry)
        industry_terms = []
        target_own_industry_lower = target_own_industry.lower()
        # Extract meaningful words (length >= 4, not common stop words)
        words = target_own_industry_lower.replace('-', ' ').replace('_', ' ').split()
        meaningful_words = [w for w in words if len(w) >= 4 and w not in ['and', 'the', 'for', 'with']]
        industry_terms.extend(meaningful_words)
        
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
            
            # Also check for key industry terms even if exact match fails (generic fuzzy matching)
            # This helps find companies in the same industry that might be classified slightly differently
            if not industry_match.get(ticker, 0):
                # Extract key terms from target's similar industries
                target_industry_terms = []
                for ind in target_similar_industries:
                    ind_lower = ind.lower()
                    words = ind_lower.replace('-', ' ').replace('_', ' ').split()
                    meaningful_words = [w for w in words if len(w) >= 4 and w not in ['and', 'the', 'for', 'with']]
                    target_industry_terms.extend(meaningful_words)
                
                # Check if candidate's industry shares key terms with target's industries
                target_has_terms = len(target_industry_terms) > 0
                candidate_has_terms = any(term in industry for term in target_industry_terms)
                
                if target_has_terms and candidate_has_terms:
                    industry_match[ticker] = 1
                    industry_hits.add(ticker)
        elif target_own_industry and industry:
            # Fallback: Check if candidate's industry matches target's own industry
            # Use fuzzy matching (contains, word overlap)
            industry_terms = [term.strip() for term in target_own_industry.split() if len(term.strip()) > 3]
            for term in industry_terms:
                if term in industry:
                    industry_match[ticker] = 1
                    industry_hits.add(ticker)
                    break
            
            # Also check for common industry type keywords (generic - extract from industry name)
            # Extract meaningful terms from target's own industry
            target_industry_words = [w for w in target_own_industry.lower().split() if len(w) >= 4]
            # Check if candidate's industry shares any meaningful terms
            for term in target_industry_words:
                if term in industry:
                    industry_match[ticker] = 1
                    industry_hits.add(ticker)
                    break
        
        # Country match
        if target_country:
            country_match[ticker] = 1 if country == target_country else 0
        else:
            country_match[ticker] = 0
    
    sector_industry_hits = sector_hits | industry_hits
    print(f"    Found {len(sector_hits)} sector matches, {len(industry_hits)} industry matches")
    
    # 5. Combine paths (before business model filter)
    prelim_candidates = semantic_hits | keyword_hits | sector_industry_hits
    print(f"  Combined: {len(prelim_candidates)} preliminary candidates")
    
    # 6. Path D - Business Model Filter (lightweight keyword-based, before LLM extraction)
    # This filters out obviously wrong business models (e.g., pure SaaS for services target)
    # Uses target's business_model_type and simple keyword matching on summaries
    print(f"  Path D: Business model filtering...")
    
    target_bm_type = target.get('business_model_type', '').lower()
    target_services_share = target.get('services_share_estimate', 0.5)
    
    # Keywords that indicate services vs software/product companies
    services_keywords = [
        'consulting', 'advisory', 'professional services', 'implementation',
        'integration', 'managed services', 'outsourcing', 'business process services',
        'staff augmentation', 'time and materials', 'project-based', 'advisory services'
    ]
    software_keywords = [
        'saas', 'subscription software', 'software platform', 'our platform',
        'license fees', 'perpetual license', 'licensed software', 'api platform',
        'software-as-a-service', 'cloud platform', 'software product'
    ]
    marketplace_keywords = ['marketplace', 'platform marketplace', 'two-sided marketplace']
    hardware_keywords = ['hardware', 'semiconductor', 'equipment manufacturing']
    financial_keywords = ['bank', 'financial institution', 'insurance company', 'lending']
    
    business_model_filtered = set()
    business_model_rejected = set()
    
    # Only apply business model filter if target has business model info
    if target_bm_type or target_services_share is not None:
        # Determine what to filter for
        is_services_target = (
            target_bm_type in ['services', 'hybrid_services_software'] or
            (target_services_share is not None and target_services_share >= 0.5)
        )
        
        # Disallowed types (from config, but hardcode for prelim since we don't have LLM extraction yet)
        disallowed_types = ['marketplace', 'hardware', 'financial_institution', 'other']
        
        for ticker in prelim_candidates:
            row = universe_df[universe_df['ticker'] == ticker].iloc[0] if len(universe_df[universe_df['ticker'] == ticker]) > 0 else None
            if row is None:
                continue
            
            summary_lower = str(row.get('summary', '')).lower()
            industry_lower = str(row.get('industry', '')).lower()
            
            # Quick keyword-based business model detection (lightweight, no LLM)
            services_hits = sum(1 for kw in services_keywords if kw in summary_lower or kw in industry_lower)
            software_hits = sum(1 for kw in software_keywords if kw in summary_lower or kw in industry_lower)
            marketplace_hits = sum(1 for kw in marketplace_keywords if kw in summary_lower or kw in industry_lower)
            hardware_hits = sum(1 for kw in hardware_keywords if kw in summary_lower or kw in industry_lower)
            financial_hits = sum(1 for kw in financial_keywords if kw in summary_lower or kw in industry_lower)
            
            # Reject disallowed types
            if marketplace_hits >= 2 or hardware_hits >= 2 or financial_hits >= 2:
                business_model_rejected.add(ticker)
                continue
            
            # If target is services-focused, penalize/reject pure software companies
            if is_services_target:
                # If clearly software-heavy (3+ software keywords, 1 or fewer services keywords)
                if software_hits >= 3 and services_hits <= 1:
                    # Reject pure software companies for services target
                    business_model_rejected.add(ticker)
                    continue
                # If balanced or services-heavy, keep
                business_model_filtered.add(ticker)
            else:
                # Target is not services-focused, keep all (or apply reverse logic if needed)
                business_model_filtered.add(ticker)
    
    # Apply business model filter
    if business_model_filtered:
        prelim_candidates = prelim_candidates & business_model_filtered
        print(f"    Business model filter: kept {len(business_model_filtered)}, rejected {len(business_model_rejected)}")
    else:
        # If no business model info in target, skip filtering
        print(f"    Business model filter: skipped (no target business model info)")
    
    print(f"  Final (after business model filter): {len(prelim_candidates)} preliminary candidates")
    
    # 7. Compute normalized scores and score_pre
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

