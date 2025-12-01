"""
run_pipeline.py: Main orchestrator for the company comparator pipeline.

Supports two modes:
1. Use existing target.json: --target data/target.json
2. Create target.json from basic info: --name, --url, --description, --primary-industry-classification
"""
# CRITICAL: Set environment variables BEFORE any imports to prevent segfaults
# Fix OpenMP conflict on macOS (XGBoost/FAISS/NumPy all use OpenMP)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Disable NUMBA JIT to avoid numba initialization issues with shap on some macOS/Python combos
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
# Additional safeguards for macOS
os.environ.setdefault('OMP_NUM_THREADS', '1')  # Limit OpenMP threads

import sys
import json
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

# Import modules
from universe.generate_candidates import generate_candidates
from prelim.prelim_filter import prelim_filter
from evidence.pack import build_evidence_pack
from nlp.llm_extract import extract_llm_structured
from features.product_overlap import score_product_overlap
from features.customer_overlap import score_customer_overlap
# M feature removed - redundant with S (both measure segment similarity)
# from features.segment_mix import score_segment_mix
from features.semantic import score_semantic_similarity  # Fallback only
# Import embedding function from universe module
from universe.embeddings_index import get_cached_embedding
from features.industry_prox import score_industry_proximity
from features.evidence_quality import score_evidence_quality
from features.recency import score_recency
from ranker.scorer_rule import rule_score
# Import export_csv directly (avoiding conflict with built-in 'io' module)
import importlib.util
export_csv_path = os.path.join(ROOT, 'src', 'io', 'export_csv.py')
spec = importlib.util.spec_from_file_location("export_csv", export_csv_path)
export_csv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_csv)
export_leaderboard = export_csv.export_leaderboard

# Paths
CONFIG_DIR = os.path.join(ROOT, 'config')
DATA_DIR = os.path.join(ROOT, 'data')
OUTPUTS_DIR = os.path.join(DATA_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Add config to path (for scoring_config imports)
sys.path.insert(0, CONFIG_DIR)


def load_config():
    """Load runtime configuration."""
    runtime_path = os.path.join(CONFIG_DIR, 'runtime.yaml')
    if os.path.exists(runtime_path):
        with open(runtime_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_target(target_path):
    """Load target JSON."""
    with open(target_path, 'r') as f:
        return json.load(f)


def _map_evidence_to_features(extracted, pack, target, candidate_row=None):
    """
    Map LLM evidence to specific features (P, C, M, S).
    
    Returns:
        dict with keys 'P', 'C', 'M', 'S', each containing list of evidence dicts
    """
    evidence_by_feature = {
        'P': [],  # Product similarity evidence
        'C': [],  # Customer similarity evidence
        'M': [],  # Product mix evidence
        'S': []   # Semantic similarity evidence (general)
    }
    
    # Get LLM-extracted evidence
    llm_evidence = extracted.get('evidence', [])
    
    # Extract product-related evidence (for Feature P)
    target_products = target.get('products', []) + target.get('business_activity', [])
    if isinstance(llm_evidence, list):
        for quote_obj in llm_evidence:
            quote_text = (quote_obj.get('quote', '') or '').lower()
            # Check if quote mentions products/services
            if any(prod.lower() in quote_text for prod in target_products if prod):
                evidence_by_feature['P'].append({
                    'quote': quote_obj.get('quote', '')[:300],  # Limit length
                    'source': quote_obj.get('source', 'llm_extraction'),
                    'source_url': quote_obj.get('source_url', ''),
                    'category': 'products'
                })
    elif isinstance(llm_evidence, dict):
        # Evidence organized by category
        for category in ['products', 'business_activity']:
            category_quotes = llm_evidence.get(category, [])
            if isinstance(category_quotes, list):
                for quote_obj in category_quotes[:2]:  # Max 2 per category
                    if isinstance(quote_obj, dict) and quote_obj.get('quote'):
                        evidence_by_feature['P'].append({
                            'quote': quote_obj.get('quote', '')[:300],
                            'source': quote_obj.get('source', 'llm_extraction'),
                            'source_url': quote_obj.get('source_url', ''),
                            'category': 'products'
                        })
    
    # Extract customer-related evidence (for Feature C)
    target_customers = target.get('customer_segment', target.get('customers', []))
    if isinstance(llm_evidence, list):
        for quote_obj in llm_evidence:
            quote_text = (quote_obj.get('quote', '') or '').lower()
            # Check if quote mentions customers/segments
            if any(cust.lower() in quote_text for cust in target_customers if cust):
                evidence_by_feature['C'].append({
                    'quote': quote_obj.get('quote', '')[:300],
                    'source': quote_obj.get('source', 'llm_extraction'),
                    'source_url': quote_obj.get('source_url', ''),
                    'category': 'customers'
                })
    elif isinstance(llm_evidence, dict):
        customer_quotes = llm_evidence.get('customer_segment', llm_evidence.get('customers', []))
        if isinstance(customer_quotes, list):
            for quote_obj in customer_quotes[:2]:
                if isinstance(quote_obj, dict) and quote_obj.get('quote'):
                    evidence_by_feature['C'].append({
                        'quote': quote_obj.get('quote', '')[:300],
                        'source': quote_obj.get('source', 'llm_extraction'),
                        'source_url': quote_obj.get('source_url', ''),
                        'category': 'customers'
                    })
    
    # Extract product mix evidence (for Feature M)
    target_mix = target.get('product_mix', {})
    candidate_mix = extracted.get('segment_mix', {})
    if target_mix and candidate_mix:
        # Look for revenue/segment mentions
        sources = pack.get('sources', [])
        for source in sources:
            text = (source.get('text', '') or '').lower()
            if any(keyword in text for keyword in ['revenue', 'segment', 'business unit', '%', 'percent']):
                evidence_by_feature['M'].append({
                    'quote': source.get('text', '')[:300],
                    'source': source.get('type', 'unknown'),
                    'source_url': source.get('url', ''),
                    'category': 'product_mix'
                })
                if len(evidence_by_feature['M']) >= 2:
                    break
    
    # Semantic similarity evidence (for Feature S)
    # First, try to get Path B semantic evidence from candidate data (NLP embeddings)
    if candidate_row is not None:
        # Check for Path B semantic evidence stored in candidate_row
        path_b_evidence_str = candidate_row.get('path_b_evidence', '{}')
        if path_b_evidence_str and path_b_evidence_str != '{}':
            try:
                import json
                path_b_evidence = json.loads(path_b_evidence_str) if isinstance(path_b_evidence_str, str) else path_b_evidence_str
                if path_b_evidence and path_b_evidence.get('method') == 'nlp_embedding':
                    # Add Path B semantic evidence for explainability
                    summary_text = path_b_evidence.get('summary_text', '')
                    similarity = path_b_evidence.get('similarity', 0.0)
                    target_keywords = path_b_evidence.get('target_keywords', '')
                    
                    if summary_text:
                        # Create evidence quote showing semantic match
                        evidence_quote = f"Matched via semantic keyword search (similarity {similarity:.2f}). Company summary mentions concepts similar to target keywords: '{target_keywords}'. Summary excerpt: {summary_text[:200]}..."
                        evidence_by_feature['S'].append({
                            'quote': evidence_quote[:300],
                            'source': 'path_b_nlp_embedding',
                            'source_url': '',
                            'category': 'semantic_keyword_match',
                            'similarity_score': similarity,
                            'target_keywords': target_keywords,
                            'summary_excerpt': summary_text[:200]
                        })
            except Exception as e:
                # If parsing fails, continue without Path B evidence
                pass
        
        # Also check for path_b_summary in candidate_row (from generate_candidates)
        path_b_summary = candidate_row.get('path_b_summary', '')
        path_b_similarity = candidate_row.get('path_b_similarity', None)
        path_b_target_keywords = candidate_row.get('path_b_target_keywords', '')
        
        if path_b_summary and path_b_similarity is not None:
            # Add Path B evidence from generate_candidates
            evidence_quote = f"Matched via semantic keyword search (similarity {path_b_similarity:.2f}). Company summary mentions concepts similar to target keywords: '{path_b_target_keywords}'. Summary excerpt: {path_b_summary[:200]}..."
            evidence_by_feature['S'].append({
                'quote': evidence_quote[:300],
                'source': 'path_b_nlp_embedding',
                'source_url': '',
                'category': 'semantic_keyword_match',
                'similarity_score': path_b_similarity,
                'target_keywords': path_b_target_keywords,
                'summary_excerpt': path_b_summary[:200]
            })
    
    # Add LLM-extracted semantic evidence
    if isinstance(llm_evidence, list) and len(llm_evidence) > 0:
        # Use first general evidence quote
        first_quote = llm_evidence[0]
        if isinstance(first_quote, dict) and first_quote.get('quote'):
            evidence_by_feature['S'].append({
                'quote': first_quote.get('quote', '')[:300],
                'source': first_quote.get('source', 'llm_extraction'),
                'source_url': first_quote.get('source_url', ''),
                'category': 'semantic'
            })
    
    # Limit each feature to max 2 evidence quotes
    for feature in ['P', 'C', 'M', 'S']:
        evidence_by_feature[feature] = evidence_by_feature[feature][:2]
    
    return evidence_by_feature


def _build_natural_language_explanation(ranked_row, evidence_by_feature, score_linear):
    """
    Build a natural language explanation that includes SHAP values AND evidence quotes.
    
    Returns a human-readable string like:
    "Ranked #4 with score 0.78. Product similarity was a positive driver (+0.12).
    Evidence: 'We provide healthcare analytics, ERP implementation...'
    Customer similarity contributed (+0.08). Evidence: 'Our main clients include hospitals...'
    Penalties: mix (-0.03)."
    """
    parts = []
    
    rank = int(ranked_row.get('rank_ml', 0))
    score = float(ranked_row.get('score_linear', score_linear))
    
    # Overall ranking
    parts.append(f"Ranked #{rank} with score {score:.2f}.")
    
    # Business model classification (critical for services firms)
    bm_type = ranked_row.get('business_model_type', 'unknown')
    services_share = float(ranked_row.get('services_share_estimate', 0.5) or 0.5)
    penalty = float(ranked_row.get('penalty_producty', 0.0) or 0.0)
    bm_gate = ranked_row.get('gate_business_model', False)
    
    if bm_type:
        bm_label = {
            'services': 'services-heavy',
            'software': 'software-led',
            'hybrid_services_software': 'hybrid (services + software)',
            'marketplace': 'marketplace',
            'hardware': 'hardware',
            'financial_institution': 'financial institution',
            'other': 'other'
        }.get(bm_type.lower(), bm_type)
        
        services_pct = int(services_share * 100)
        parts.append(f"Classified as **{bm_label}** (estimated ~{services_pct}% of revenue from services).")
        
        if penalty > 0.01:
            parts.append(f"Penalty applied due to low services share (penalty: -{penalty:.2f}).")
        elif services_share >= 0.7:
            parts.append(f"No penalty from software weighting (services_share_estimate = {services_share:.1f}).")
    
    if not bm_gate:
        parts.append(f"**Note: Did not pass business model gate** (requires services_share >= 0.5 and services-focused business model).")
    
    # Positive drivers (SHAP > 0) with evidence
    shap_p = float(ranked_row.get('shap_P', 0.0)) if not pd.isna(ranked_row.get('shap_P', np.nan)) else 0.0
    shap_c = float(ranked_row.get('shap_C', 0.0)) if not pd.isna(ranked_row.get('shap_C', np.nan)) else 0.0
    shap_m = float(ranked_row.get('shap_M', 0.0)) if not pd.isna(ranked_row.get('shap_M', np.nan)) else 0.0
    shap_s = float(ranked_row.get('shap_S', 0.0)) if not pd.isna(ranked_row.get('shap_S', np.nan)) else 0.0
    
    # Product similarity (P)
    if shap_p > 0.01:  # Only mention if meaningful contribution
        evidence_p = evidence_by_feature.get('P', [])
        evidence_text = ""
        if evidence_p and len(evidence_p) > 0:
            quote = evidence_p[0].get('quote', '')
            if quote:
                # Truncate to ~140 chars for readability
                truncated = quote[:140] + "..." if len(quote) > 140 else quote
                evidence_text = f" Evidence: '{truncated}'"
        parts.append(f"Product similarity was a positive driver (+{shap_p:.2f}).{evidence_text}")
    
    # Customer similarity (C)
    if shap_c > 0.01:
        evidence_c = evidence_by_feature.get('C', [])
        evidence_text = ""
        if evidence_c and len(evidence_c) > 0:
            quote = evidence_c[0].get('quote', '')
            if quote:
                truncated = quote[:140] + "..." if len(quote) > 140 else quote
                evidence_text = f" Evidence: '{truncated}'"
        parts.append(f"Customer-segment similarity contributed (+{shap_c:.2f}).{evidence_text}")
    
    # Product mix (M)
    if shap_m > 0.01:
        evidence_m = evidence_by_feature.get('M', [])
        evidence_text = ""
        if evidence_m and len(evidence_m) > 0:
            quote = evidence_m[0].get('quote', '')
            if quote:
                truncated = quote[:140] + "..." if len(quote) > 140 else quote
                evidence_text = f" Evidence: '{truncated}'"
        parts.append(f"Product-mix alignment added (+{shap_m:.2f}).{evidence_text}")
    
    # Semantic similarity (S) - usually smaller contribution
    if shap_s > 0.01:
        parts.append(f"Semantic similarity improved ranking (+{shap_s:.2f}).")
    
    # Penalties (negative SHAP values)
    penalties = []
    if shap_p < -0.01:
        penalties.append(f"products ({shap_p:.2f})")
    if shap_c < -0.01:
        penalties.append(f"customers ({shap_c:.2f})")
    if shap_m < -0.01:
        penalties.append(f"mix ({shap_m:.2f})")
    if shap_s < -0.01:
        penalties.append(f"semantic ({shap_s:.2f})")
    
    if penalties:
        parts.append("Penalties: " + ", ".join(penalties) + ".")
    
    # Fallback if no SHAP values available (use rule-based breakdown)
    if pd.isna(ranked_row.get('shap_P', np.nan)):
        p_value = float(ranked_row.get('P', 0.0))
        c_value = float(ranked_row.get('C', 0.0))
        m_value = float(ranked_row.get('M', 0.0))
        s_value = float(ranked_row.get('S', 0.0))
        
        if p_value > 0.3:
            evidence_p = evidence_by_feature.get('P', [])
            evidence_text = ""
            if evidence_p and len(evidence_p) > 0:
                quote = evidence_p[0].get('quote', '')
                if quote:
                    truncated = quote[:140] + "..." if len(quote) > 140 else quote
                    evidence_text = f" Evidence: '{truncated}'"
            parts.append(f"Strong product similarity ({p_value:.2f}).{evidence_text}")
        
        if c_value > 0.3:
            evidence_c = evidence_by_feature.get('C', [])
            evidence_text = ""
            if evidence_c and len(evidence_c) > 0:
                quote = evidence_c[0].get('quote', '')
                if quote:
                    truncated = quote[:140] + "..." if len(quote) > 140 else quote
                    evidence_text = f" Evidence: '{truncated}'"
            parts.append(f"Strong customer similarity ({c_value:.2f}).{evidence_text}")
    
    return " ".join(parts)


def compute_features(target, candidate_row, extracted_data, evidence_pack, run_with_openai=False):
    """Compute all features (P, C, M, S, I, E, R) for a candidate."""
    # P: Product overlap - compare business_activity to company description (summary)
    # ONLY use business_activity - no products field, no fallback
    # Compare business_activity to company descriptions (summary) in universe
    target_products = target.get('business_activity', [])
    if not target_products:
        # No fallback - if business_activity is missing, use empty list
        target_products = []
    
    # Get candidate company description (summary) from universe row
    candidate_summary = str(candidate_row.get('summary', ''))
    
    # Use compute_product_similarity_fast which compares target business_activity to candidate summary
    from universe.rank_key_semantic import compute_product_similarity_fast
    P = compute_product_similarity_fast(
        target_products,
        candidate_summary,
        run_with_openai=run_with_openai
    )
    
    # For backward compatibility, compute product_hits and concept_matches
    # (simplified - just count keyword matches in summary)
    product_hits = 0
    concept_matches = []
    if target_products and candidate_summary:
        candidate_lower = candidate_summary.lower()
        for tp in target_products:
            if tp.lower() in candidate_lower:
                product_hits += 1
                concept_matches.append({
                    'concept': tp,
                    'match_type': 'keyword',
                    'materiality_0_1': 1.0,
                    'match_strength': 1.0
                })
    
    # S: Segment distribution similarity (portfolio-aware, replaces semantic similarity and M)
    # NOTE: M feature removed - it was redundant with S (both measure segment similarity)
    # S = cosine(segment_mix) * (1 - λ * entropy_penalty) is more sophisticated than M = cosine(segment_mix)
    # Uses segment_mix distributions with cosine similarity + entropy mismatch penalty
    from features.segment_distribution import compute_segment_s_score
    
    # Get vocabulary if precomputed (for efficiency)
    vocabulary = candidate_row.get('segment_vocabulary', None)
    lambda_entropy = 0.4  # Tuning parameter (only used when target is diversified)
    diversification_threshold = 0.3  # Entropy threshold to consider target "diversified"
    
    seg_s_result = None  # Initialize for reuse in gates and C feature
    try:
        # Add candidate summary to extracted_data for fallback segment inference
        # This helps when LLM extraction returns empty customer_segment
        candidate_summary = str(candidate_row.get('summary', ''))
        if candidate_summary and 'summary' not in extracted_data:
            extracted_data_with_summary = extracted_data.copy()
            extracted_data_with_summary['summary'] = candidate_summary
        else:
            extracted_data_with_summary = extracted_data
        
        seg_s_result = compute_segment_s_score(
            target,
            extracted_data_with_summary,
            vocabulary=vocabulary,
            lambda_entropy=lambda_entropy,
            diversification_threshold=diversification_threshold
        )
        S = seg_s_result.get('S', 0.0)
        # Store detailed fields for explainability
        seg_s_details = {
            'sim_cosine': seg_s_result.get('sim_cosine', 0.0),
            'entropy_target': seg_s_result.get('entropy_target', 0.0),
            'entropy_candidate': seg_s_result.get('entropy_candidate', 0.0),
            'penalty_entropy': seg_s_result.get('penalty_entropy', 0.0),
            'segment_mix_target': seg_s_result.get('segment_mix_target', {}),
            'segment_mix_candidate': seg_s_result.get('segment_mix_candidate', {})
        }
    except Exception as e:
        # Fallback to old semantic similarity if segment distribution fails
        print(f"    Warning: Segment distribution S failed: {e}, using semantic fallback")
        target_text = target.get('raw_profile_text') or target.get('text_profile', '')
        if not target_text:
            target_products = target.get('business_activity', []) or target.get('products', [])
            target_customers = target.get('customer_segment', []) or target.get('customers', [])
            target_text = ' '.join(target_products + target_customers)
        candidate_text = candidate_row.get('summary', '') or ''
        
        # Use semantic similarity as fallback
        from features.semantic import score_semantic_similarity
        from universe.embeddings_index import get_cached_embedding
        
        try:
            target_emb = get_cached_embedding(target_text, run_with_openai=run_with_openai)
            candidate_emb = get_cached_embedding(candidate_text[:2000], run_with_openai=run_with_openai)
            S = score_semantic_similarity(target_emb, candidate_emb) if target_emb and candidate_emb else 0.0
            seg_s_details = {
                'sim_cosine': S,
                'entropy_target': 0.0,
                'entropy_candidate': 0.0,
                'penalty_entropy': 0.0,
                'segment_mix_target': {},
                'segment_mix_candidate': {}
            }
        except Exception:
            S = 0.0
            seg_s_details = {
                'sim_cosine': 0.0,
                'entropy_target': 0.0,
                'entropy_candidate': 0.0,
                'penalty_entropy': 0.0,
                'segment_mix_target': {},
                'segment_mix_candidate': {}
            }
    
    # C: Customer overlap
    # Option 1: Use segment_mix similarity from S feature (recommended - leverages sophisticated segment distribution)
    # Option 2: Use enhanced customer overlap with semantic embeddings
    # Option 3: Fallback to customer_industries matching (when customer_segment is at different abstraction levels)
    # Option 4: Fallback to substring matching
    target_customers = target.get('customer_segment', target.get('customers', []))
    candidate_customers_raw = extracted_data.get('customer_segment', [])
    # Filter generic phrases
    from nlp.llm_extract import _filter_generic_phrases
    candidate_customers = _filter_generic_phrases(candidate_customers_raw)
    
    # If all were filtered out, try primary_customer_types or customer_industries
    if not candidate_customers and candidate_customers_raw:
        # Try primary_customer_types (new field from LLM)
        primary_customers = extracted_data.get('primary_customer_types', [])
        if primary_customers:
            candidate_customers = _filter_generic_phrases(primary_customers)
        # Fallback to customer_industries if still empty
        if not candidate_customers:
            customer_industries = extracted_data.get('customer_industries', [])
            if customer_industries:
                candidate_customers = _filter_generic_phrases(customer_industries)
    
    # Try to use segment_mix similarity first (from S feature computation above)
    # This avoids redundancy and leverages the more sophisticated segment distribution similarity
    C = 0.0
    customer_hits = 0
    if seg_s_result and seg_s_result.get('sim_cosine', 0.0) > 0 and seg_s_result.get('segment_mix_target') and seg_s_result.get('segment_mix_candidate'):
        # Option 1: Use segment distribution similarity (recommended - leverages S feature)
        from features.customer_overlap import score_customer_overlap_with_segment_mix
        C_segment, customer_hits_segment = score_customer_overlap_with_segment_mix(
            target, extracted_data, seg_s_result
        )
        # Use segment-based score if meaningful
        if C_segment > 0.05:  # Use if meaningful (lower threshold since it's already normalized)
            C = C_segment
            customer_hits = customer_hits_segment
        else:
            # Fallback to direct customer overlap with embeddings
            C, customer_hits = score_customer_overlap(
                target_customers, candidate_customers, 
                run_with_openai=run_with_openai, similarity_threshold=0.6
            )
    else:
        # Option 2: Use enhanced customer overlap with semantic embeddings
        C, customer_hits = score_customer_overlap(
            target_customers, candidate_customers, 
            run_with_openai=run_with_openai, similarity_threshold=0.6
        )
    
    # ENHANCEMENT: If customer_segment matching gives low/zero score, try customer_industries as fallback
    # This handles cases where customer_segment descriptions are at different abstraction levels
    # (e.g., Cargill: "farmers" vs ADM: "food industry" - both serve agriculture/food verticals)
    if C < 0.3:  # If customer_segment matching gave low score
        target_customer_industries = target.get('customer_industries', [])
        candidate_customer_industries = extracted_data.get('customer_industries', [])
        
        if target_customer_industries and candidate_customer_industries:
            # Try matching on customer_industries (verticals served) instead
            # Note: score_customer_overlap is already imported at top of file
            C_industries, customer_hits_industries = score_customer_overlap(
                target_customer_industries, candidate_customer_industries,
                run_with_openai=run_with_openai, similarity_threshold=0.5  # Lower threshold for industry matching
            )
            
            # Use the higher of the two scores (customer_segment or customer_industries)
            # This ensures we capture overlap even when descriptions are at different abstraction levels
            if C_industries > C:
                C = C_industries
                customer_hits = customer_hits_industries
    
    # NOTE: S feature computation moved earlier (before C feature) to enable consolidation
    # S and seg_s_details are already computed above (around line 404)
    
    # I: Industry proximity
    # CRITICAL: Compare customer industries (verticals served), not own industry
    # Both target and candidate should be in the same own industry (filtered earlier)
    # We compare which customer industries each serves
    
    target_own_industry = target.get('primary_industry_classification', '').lower()
    candidate_own_industry = str(candidate_row.get('industry', '')).lower()
    
    # SIC industry bonus (small hint, not a filter)
    # Normalize industry strings for comparison
    def normalize_industry(industry_str):
        """Normalize industry string for comparison."""
        if not industry_str:
            return ''
        return str(industry_str).lower().strip()
    
    target_sic_norm = normalize_industry(target_own_industry)
    candidate_sic_norm = normalize_industry(candidate_own_industry)
    same_sic = 1.0 if target_sic_norm and candidate_sic_norm and target_sic_norm == candidate_sic_norm else 0.0
    
    # Get customer industries (verticals served) from both target and candidate
    target_customer_industries = target.get('customer_industries', target.get('industries', []))  # Backward compatibility: fallback to old 'industries' field
    candidate_customer_industries = extracted_data.get('customer_industries', extracted_data.get('industries', []))  # From candidate's LLM extraction
    
    # First check: Are they in the same own industry? (e.g., both consulting firms)
    # If not, low score
    if target_own_industry and candidate_own_industry:
        # More precise industry type matching (avoid overly broad "services" match)
        def categorize_industry_type(industry_str):
            """Categorize industry into broad types for matching."""
            industry_lower = industry_str.lower()
            # Consulting/Professional Services
            if any(kw in industry_lower for kw in ['consulting', 'advisory', 'professional services']):
                return 'consulting'
            # Technology/Software
            elif any(kw in industry_lower for kw in ['software', 'technology', 'tech', 'saas', 'cloud']):
                return 'technology'
            # Financial Services
            elif any(kw in industry_lower for kw in ['financial services', 'banking', 'insurance', 'fintech']):
                return 'financial'
            # Healthcare
            elif any(kw in industry_lower for kw in ['healthcare', 'health', 'medical', 'pharmaceutical']):
                return 'healthcare'
            # Manufacturing/Industrial
            elif any(kw in industry_lower for kw in ['manufacturing', 'industrial', 'automotive']):
                return 'manufacturing'
            # Retail/Consumer
            elif any(kw in industry_lower for kw in ['retail', 'consumer', 'e-commerce']):
                return 'retail'
            # Generic "services" (only if no other category matches)
            elif 'services' in industry_lower:
                return 'services'
            else:
                return 'other'
        
        target_category = categorize_industry_type(target_own_industry)
        candidate_category = categorize_industry_type(candidate_own_industry)
        
        if target_category != candidate_category:
            # Different industry types - low score
            I = 0.3
        elif target_customer_industries and candidate_customer_industries:
            # Both are in same own industry type - compare customer industries served
            # Calculate Jaccard similarity (overlap / union)
            target_set = set([ind.lower().strip() for ind in target_customer_industries if ind])
            candidate_set = set([ind.lower().strip() for ind in candidate_customer_industries if ind])
            
            if target_set and candidate_set:
                overlap = len(target_set & candidate_set)
                union = len(target_set | candidate_set)
                I = overlap / union if union > 0 else 0.0  # Jaccard similarity
            else:
                I = 0.5  # Missing data
        elif target_customer_industries or candidate_customer_industries:
            # One has customer industries, other doesn't - partial score
            I = 0.4
        else:
            # Both in same own industry but no customer industry data - neutral
            # Fall back to traditional industry proximity if available
            if target_own_industry and candidate_own_industry:
                I = score_industry_proximity(target_own_industry, candidate_own_industry)
            else:
                I = 0.5
    else:
        # Missing own industry data - fall back to traditional method
        target_industry = target.get('industry', '') or target_own_industry
        candidate_industry = candidate_row.get('industry', '')
        I = score_industry_proximity(target_industry, candidate_industry) if target_industry else 0.5
    
    # V: Vertical similarity (multi-hot encoding of vertical categories)
    # Encodes companies as multi-hot vectors of verticals they serve (healthcare, education, gov, etc.)
    # Uses cosine similarity instead of matching generic terms like "consulting"
    from features.vertical_similarity import score_vertical_similarity
    V = score_vertical_similarity(
        target_data=target,
        candidate_data=extracted_data,
        candidate_text=candidate_summary
    )
    
    # E: Evidence quality
    sources = evidence_pack.get('sources', [])
    E = score_evidence_quality(sources)
    
    # R: Recency
    updated_at = evidence_pack.get('updated_at', '')
    R = score_recency(updated_at)
    
    # B: Business model similarity (legacy - still computed for backward compatibility)
    from features.business_model_similarity import business_model_similarity
    B = business_model_similarity(target, extracted_data)
    
    # E_SIG: Economic signature similarity (NEW - universal economic structure matching)
    # Compares HOW companies make money (revenue structure, IP, lock-in, cycles)
    # Works universally across all industries - not dependent on customer segments
    try:
        from features.economic_model_similarity import compute_economic_model_similarity_from_extracted
        target_extracted = target.get('extracted_data', {}) or {}
        E_SIG = compute_economic_model_similarity_from_extracted(target_extracted, extracted_data)
    except Exception as e:
        # Fallback if economic signature computation fails
        print(f"    Warning: Economic signature similarity computation failed: {e}")
        E_SIG = 0.5  # Default to neutral similarity
    
    # Confidence (blended: LLM confidence + evidence coverage + E score)
    llm_confidence = extracted_data.get('confidence_0_1', 0.0)
    evidence_coverage = min(len(sources) / 5.0, 1.0)  # Normalize to [0, 1]
    confidence_final = 0.4 * llm_confidence + 0.3 * evidence_coverage + 0.3 * E
    
    # Compute base linear score ONCE (reuse everywhere, avoid redundant computation)
    from ranker.scorer_rule import compute_base_score
    from scoring_config import SCORING_CONFIG
    features_dict = {
        'P': P,
        'C': C,
        'S': S,  # M removed - redundant with S
        'B': B,  # Legacy: Business model similarity (services_share based)
        'V': V,  # NEW: Vertical similarity (multi-hot encoding of vertical categories)
        'E_SIG': E_SIG,  # NEW: Economic signature similarity (universal economic structure matching)
        'I': I,
        'E': E,
        'R': R,
        'same_sic': same_sic
    }
    score_linear = compute_base_score(features_dict, SCORING_CONFIG)
    
    # Store segment S details for reuse in gates (avoid recomputing)
    seg_s_result_for_gates = seg_s_result if 'seg_s_result' in locals() else None
    
    return {
        'P': P,
        'C': C,
        'S': S,  # Segment distribution similarity (portfolio-aware, replaces M)
        'B': B,  # Legacy: Business model similarity (services_share, business_model_type, capabilities)
        'V': V,  # Vertical similarity (multi-hot encoding of vertical categories)
        'E_SIG': E_SIG,  # NEW: Economic signature similarity (universal economic structure matching)
        'I': I,
        'E': E,
        'R': R,
        'same_sic': same_sic,  # SIC industry bonus (0.0 or 1.0)
        'product_hits': product_hits,
        'customer_hits': customer_hits,
        'LLM_confidence': llm_confidence,
        'confidence_final': confidence_final,
        'concept_matches': concept_matches,
        'segment_mix': extracted_data.get('segment_mix', {}),
        'initiatives': extracted_data.get('initiatives', []),
        # Precomputed scores (reuse to avoid redundant computation)
        'score_linear': score_linear,
        # Segment distribution S details (for explainability and gate reuse)
        'sim_cosine': seg_s_details.get('sim_cosine', 0.0),
        'entropy_target': seg_s_details.get('entropy_target', 0.0),
        'entropy_candidate': seg_s_details.get('entropy_candidate', 0.0),
        'penalty_entropy': seg_s_details.get('penalty_entropy', 0.0),
        'segment_mix_target': seg_s_details.get('segment_mix_target', {}),
        'segment_mix_candidate': seg_s_details.get('segment_mix_candidate', {}),
        # Store full seg_s_result for gate reuse (avoid recomputing)
        '_seg_s_result': seg_s_result_for_gates
    }


def main():
    parser = argparse.ArgumentParser(
        description='Company Comparator Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Option 1: Use existing target.json
  python cli/run_pipeline.py --target data/target.json --openai
  
  # Option 2: Create target.json from basic info and run pipeline
  python cli/run_pipeline.py \\
    --name "Company Name" \\
    --url "https://company.com" \\
    --description "Business description..." \\
    --primary-industry-classification "Industry Name" \\
    --openai
  
  # Option 2 with LinkedIn:
  python cli/run_pipeline.py \\
    --name "Company Name" \\
    --url "https://company.com" \\
    --description "Business description..." \\
    --primary-industry-classification "Industry Name" \\
    --linkedin-url "https://linkedin.com/company/company-name" \\
    --openai
        """
    )
    
    # Target input options: either provide target.json OR provide basic info to create it
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--target', type=str, help='Path to existing target.json')
    target_group.add_argument('--name', type=str, help='Company name (creates target.json)')
    
    # Required when creating target.json
    parser.add_argument('--url', type=str, help='Company homepage URL (required if --name provided)')
    parser.add_argument('--description', type=str, help='Business description (required if --name provided)')
    parser.add_argument('--primary-industry-classification', '--industry', type=str, dest='primary_industry_classification',
                        help='Primary industry classification (required if --name provided)')
    
    # Optional for target creation
    parser.add_argument('--linkedin-url', type=str, help='LinkedIn company URL (optional)')
    parser.add_argument('--linkedin', type=str, help='LinkedIn company name/handle (optional)')
    parser.add_argument('--months-back', type=int, default=8, help='Months of LinkedIn posts to fetch (default: 8)')
    parser.add_argument('--ticker', type=str, help='Company ticker symbol (optional, for excluding target from results)')
    
    # Pipeline options
    parser.add_argument('--openai', action='store_true', help='Use real OpenAI embeddings and LLM')
    parser.add_argument('--limit-candidates', type=int, default=None, help='Limit number of candidates for testing')
    parser.add_argument('--force', action='store_true', help='Force recreation of target.json even if cached version exists')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP computation (use if experiencing segfaults)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Company Comparator Pipeline")
    print("="*80)
    
    # 0. Create target.json if basic info provided (or reuse if exists)
    target_path = args.target
    if args.name:
        # Check if target.json already exists for this company
        if not args.url or not args.description or not args.primary_industry_classification:
            parser.error("When using --name, --url, --description, and --primary-industry-classification are required")
        
        # Generate filename from company name
        safe_name = args.name.replace(' ', '_').replace('/', '_').lower()
        target_path = os.path.join(DATA_DIR, f'target_{safe_name}.json')
        
        # Check if target.json already exists (unless --force is set)
        if os.path.exists(target_path) and not args.force:
            print("\n[0/10] Loading existing target.json from cache...")
            print(f"✓ Found cached target.json: {target_path}")
            print(f"  Company: {args.name}")
            print(f"  To recreate, use --force flag")
            print()
        else:
            # Need to create target.json from basic info (or recreate with --force)
            if os.path.exists(target_path) and args.force:
                print("\n[0/10] Recreating target.json (--force flag set)...")
            else:
                print("\n[0/10] Creating target.json from input data...")
            
            # Import target creation function
            import sys
            target_creation_path = os.path.join(DATA_DIR, 'create_target_from_info.py')
            sys.path.insert(0, DATA_DIR)
            from create_target_from_info import create_target_from_info
            
            # Create target.json
            target_data = create_target_from_info(
                name=args.name,
                url=args.url,
                business_description=args.description,
                primary_industry_classification=args.primary_industry_classification,
                linkedin_url=args.linkedin_url,
                linkedin_company_name=args.linkedin,
                months_back=args.months_back,
                api_key=os.getenv('OPENAI_API_KEY') if args.openai else None,
                ticker=args.ticker
            )
            
            # Save to file
            with open(target_path, 'w') as f:
                json.dump(target_data, f, indent=2)
            print(f"✓ Created and saved target.json: {target_path}")
            print()
    
    # 1. Load configs and target
    print("\n[1/10] Loading configs and target...")
    config = load_config()
    target = load_target(target_path)
    # Sanitize target_id: remove special characters that cause path issues
    target_id = target.get('name', 'target').replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('\\', '_').replace(':', '').replace('*', '').replace('?', '').replace('"', '').replace('<', '').replace('>', '').replace('|', '').lower()
    mode = target.get('mode', 'all_segments')
    print(f"✓ Loaded target: {target.get('name')}")
    print(f"✓ Mode: {mode}")
    
    # 2. Preliminary filter (fast filter to ~200-300 candidates)
    print("\n[2/10] Running preliminary filter...")
    use_prelim_filter = config.get('use_prelim_filter', True)
    
    if use_prelim_filter:
        prelim_df = prelim_filter(target, config, run_with_openai=args.openai)
        print(f"✓ Preliminary filter: {len(prelim_df)} candidates")
        
        # Use prelim candidates for further processing
        # For now, we'll use the existing generate_candidates but could switch to prelim_df
        # For compatibility, we'll still run generate_candidates but could filter universe first
        candidates_df = generate_candidates(target, config, run_with_openai=args.openai)
        if args.limit_candidates:
            candidates_df = candidates_df.head(args.limit_candidates)
        print(f"✓ Generated {len(candidates_df)} candidates (after full candidate generation)")
    else:
        # Original path: full candidate generation
        candidates_df = generate_candidates(target, config, run_with_openai=args.openai)
        if args.limit_candidates:
            candidates_df = candidates_df.head(args.limit_candidates)
        print(f"✓ Generated {len(candidates_df)} candidates")
    
    # 3. Shortlist
    print("\n[3/10] Creating shortlist...")
    shortlist_cap = config.get('shortlist_cap', 80)
    shortlist_df = candidates_df.head(shortlist_cap).copy()
    print(f"✓ Shortlisted {len(shortlist_df)} candidates")
    
    # Display top 20 of shortlist for debugging
    print(f"\n📊 Top 20 companies in shortlist:")
    for idx, (i, row) in enumerate(shortlist_df.head(20).iterrows(), 1):
        ticker = row.get('ticker', 'N/A')
        name = row.get('name', 'N/A')[:50]
        rank_key = row.get('rank_key', 0.0)
        paths = row.get('paths', '')
        industry = str(row.get('industry', 'N/A'))[:30]
        print(f"   {idx:2d}. {ticker:6s} - {name:50s} | rank_key: {rank_key:.3f} | paths: {paths} | {industry}")
    
    # Save shortlist to file for inspection
    shortlist_path = os.path.join(OUTPUTS_DIR, f'{target_id}_shortlist.csv')
    shortlist_df.to_csv(shortlist_path, index=False)
    print(f"\n💾 Saved shortlist to: {shortlist_path}")
    
    # 4. EvidencePack gathering (real fetching with 10-K logic)
    print("\n[4/10] Gathering evidence...")
    evidence_packs = {}
    tenk_trigger_topN = config.get('tenk_trigger_topN', 30)
    total_candidates = len(shortlist_df)
    processed = 0
    failed_evidence = []
    
    for idx, row in shortlist_df.iterrows():
        ticker = row['ticker']
        cik = row.get('cik', '')
        website = row.get('website', '')
        # Handle NaN/None website values from pandas
        if pd.isna(website) or (isinstance(website, float) and np.isnan(website)):
            website = ''
        elif not isinstance(website, str):
            website = str(website).strip() if website else ''
        else:
            website = website.strip() if website else ''
        
        rank_key = row.get('rank_key', 0.0)
        paths = row.get('paths', '')
        
        # Get semantic signals if available (for smarter evidence gating)
        P_fast = row.get('P_fast', 0.0)
        C_mix = row.get('C_mix', 0.0)
        M_fast = row.get('M_fast', 0.0)
        
        # Determine if we should fetch 10-K:
        # - Top-30 by rank_key, OR
        # - Segment alias hit (path 'D'), OR
        # - High semantic signals but moderate overall rank (adjacency candidates), OR
        # - Low rank_key might indicate ambiguous evidence
        should_fetch_10k = (
            idx < tenk_trigger_topN or
            'D' in paths or
            (P_fast > 0.6 and C_mix > 0.5 and idx < 50) or  # High product + segment similarity
            (C_mix > 0.6 and M_fast > 0.5 and idx < 50) or  # High segment + model similarity
            rank_key < 0.3  # Low rank_key might indicate ambiguous evidence
        )
        
        # Smart XBRL fetching strategy:
        # - Always try to fetch XBRL (check cache first - fast path)
        # - If not cached, only fetch for top 30 (to avoid slow HTTP requests for all 80)
        # - This gives us complete data when cached, reasonable data when not cached
        should_fetch_xbrl = True  # Always try (cache check is fast)
        should_fetch_xbrl_if_not_cached = (idx < 30)  # Only fetch from API for top 30
        
        # Update config to include XBRL flags
        evidence_config = (config or {}).copy()
        evidence_config['should_fetch_xbrl'] = should_fetch_xbrl
        evidence_config['should_fetch_xbrl_if_not_cached'] = should_fetch_xbrl_if_not_cached
        
        evidence_packs[ticker] = build_evidence_pack(
            ticker=ticker,
            cik=cik,
            website=website,
            should_fetch_10k=should_fetch_10k,
            config=evidence_config
        )
        
        processed += 1
        # Print progress every 10 companies
        if processed % 10 == 0 or processed == total_candidates:
            print(f"    Progress: {processed}/{total_candidates} companies processed")
    print(f"✓ Gathered evidence for {len(evidence_packs)} candidates")
    
    # 5. LLM extraction
    print("\n[5/10] Extracting structured data with LLM...")
    
    # Set run_with_llm flag early (needed for print statements below)
    run_with_llm = args.openai  # Use LLM if OpenAI flag is set
    
    if run_with_llm:
        print("  Using OpenAI LLM for extraction (--openai flag enabled)")
        print("  ✓ Extractions will be cached to data/cache/llm_extraction/")
    else:
        print("  Using mock extraction (run with --openai for real LLM extraction)")
        print("  ⚠️  Note: Mock extractions are NOT cached. Use --openai to enable caching.")
    
    # Clear expired LLM extraction cache (non-blocking)
    try:
        from nlp.llm_extract_cache import clear_expired_extraction_cache
        cleared = clear_expired_extraction_cache()
        if cleared > 0:
            print(f"  Cleared {cleared} expired LLM extraction cache entries")
    except Exception:
        pass
    
    extracted_data = {}
    prompt_version = config.get('prompt_version', 'svc_cust_v3')
    
    def _refine_business_model(extracted, pack):
        """
        Light refinement of business model classification using keyword heuristics.
        LLM is primary classifier, but this catches obvious edge cases.
        Uses config-based classify_business_model() function - no hardcoded thresholds.
        """
        # Import config-based classification function
        # Note: os and sys are already imported globally, so don't re-import them here
        try:
            # Add both src/config and src to path for imports
            cli_dir = os.path.dirname(os.path.abspath(__file__))
            src_config_path = os.path.join(cli_dir, '../src/config')
            src_path = os.path.join(cli_dir, '../src')
            sys.path.insert(0, src_config_path)
            sys.path.insert(0, src_path)
            from config.business_model_config import classify_business_model, BM_CONFIG
        except ImportError as e:
            # Fallback: if config not available, skip refinement
            # print(f"Warning: Could not import BusinessModelConfig: {e}")  # Debug only
            return extracted
        
        # Get combined text from all sources
        website_texts = ' '.join([
            s.get('text', '') for s in pack.get('sources', [])
            if s.get('type', '').lower() in ['site', 'ir', '10k', '10-k']
        ]).lower()
        
        bm = (extracted.get('business_model_type') or 'other').lower()
        services_share = float(extracted.get('services_share_estimate', 0.5) or 0.5)
        has_software_product = extracted.get('has_software_product', False)
        
        # Keywords for services vs software (generic patterns, not company-specific)
        services_words = [
            'consulting', 'advisory', 'professional services', 'implementation',
            'integration', 'managed services', 'outsourcing', 'business process services',
            'staff augmentation', 'time and materials', 'project-based'
        ]
        software_words = [
            'saas', 'subscription software', 'software platform', 'our platform',
            'license fees', 'perpetual license', 'licensed software', 'api platform',
            'software-as-a-service', 'cloud platform', 'enterprise software'
        ]
        
        services_hits = sum(1 for w in services_words if w in website_texts)
        software_hits = sum(1 for w in software_words if w in website_texts)
        
        # Only override in very clear cases (LLM is primary)
        # Use keyword signals to adjust services_share, then use config-based classification
        if services_hits >= 3 and software_hits <= 1:
            # Very clear services firm - adjust services_share and re-classify
            services_share = max(services_share, BM_CONFIG.services_pure_min - 0.1)  # At least close to services threshold
            bm = classify_business_model(services_share, has_software_product, BM_CONFIG)
        elif software_hits >= 3 and services_hits <= 1:
            # Very clear software firm - adjust services_share and re-classify
            services_share = min(services_share, BM_CONFIG.software_pure_max + 0.1)  # At most slightly above software threshold
            bm = classify_business_model(services_share, has_software_product or True, BM_CONFIG)  # Ensure has_software_product is True
        elif software_hits >= 2 and services_hits >= 2:
            # Clear hybrid signals - adjust services_share to hybrid range and re-classify
            # Use middle of hybrid range
            hybrid_mid = (BM_CONFIG.hybrid_min + BM_CONFIG.hybrid_max) / 2.0
            services_share = (services_share + hybrid_mid) / 2.0  # Blend with hybrid midpoint
            bm = classify_business_model(services_share, has_software_product or True, BM_CONFIG)
        
        extracted['business_model_type'] = bm
        extracted['services_share_estimate'] = services_share
        extracted['services_hits'] = services_hits  # For debugging
        extracted['software_hits'] = software_hits  # For debugging
        
        return extracted
    
    failed_extractions = []
    for ticker, pack in evidence_packs.items():
        # Store base extraction first (preserve even if refinement fails)
        base_extracted = None
        try:
            extracted = extract_llm_structured(
                pack, 
                prompt_version=prompt_version,
                run_with_llm=run_with_llm,
                use_cache=True  # Explicitly enable cache - shares cache with pipeline2
            )
            # Preserve base extraction in case refinement fails
            base_extracted = extracted.copy() if extracted else None
            
            # DISABLED: _refine_business_model and config validation to avoid sys scoping errors
            # The LLM extraction already provides good classification, so refinement is optional
            # Skip refinement entirely - it uses sys.path.insert() which causes scoping errors
            refined_extracted = extracted
            
            extracted_data[ticker] = refined_extracted
        except Exception as e:
            failed_extractions.append({
                'ticker': ticker,
                'error': str(e)
            })
            print(f"    ERROR: Failed LLM extraction for {ticker}: {e}")
            # Use base extraction if available (from before refinement), otherwise try mock, otherwise empty dict
            if base_extracted:
                extracted_data[ticker] = base_extracted
            else:
                try:
                    # Try to get mock extraction as fallback
                    mock_extracted = extract_llm_structured(
                        pack,
                        prompt_version=prompt_version,
                        run_with_llm=False,  # Force mock mode
                        use_cache=False
                    )
                    if mock_extracted:
                        extracted_data[ticker] = mock_extracted
                    else:
                        extracted_data[ticker] = {}
                except Exception:
                    # If even mock fails, use empty dict
                    extracted_data[ticker] = {}
    
    if failed_extractions:
        print(f"  ⚠️  Failed LLM extraction for {len(failed_extractions)} companies:")
        for fe in failed_extractions[:10]:
            print(f"    - {fe['ticker']}: {fe['error']}")
    
    print(f"✓ Extracted data for {len(extracted_data)} candidates (expected {len(evidence_packs)})")
    
    if len(extracted_data) < len(evidence_packs):
        print(f"  ⚠️  WARNING: Only {len(extracted_data)}/{len(evidence_packs)} companies have extraction data!")
    
    # Explicitly ensure all LLM extractions are saved to cache for later inspection
    # This happens even if they were loaded from cache (refreshes metadata)
    if run_with_llm:
        cache_saved_count = 0
        cache_failed_count = 0
        try:
            from nlp.llm_extract_cache import save_cached_extraction
            for ticker, extracted in extracted_data.items():
                if ticker in evidence_packs and extracted:
                    try:
                        pack = evidence_packs[ticker]
                        saved = save_cached_extraction(ticker, pack, extracted, prompt_version)
                        if saved:
                            cache_saved_count += 1
                        else:
                            cache_failed_count += 1
                    except Exception:
                        cache_failed_count += 1
            if cache_saved_count > 0:
                print(f"  ✓ Saved {cache_saved_count} LLM extractions to cache (data/cache/llm_extraction/)")
            if cache_failed_count > 0:
                print(f"  ⚠️  Failed to cache {cache_failed_count} extractions (non-critical)")
            if cache_saved_count == 0 and cache_failed_count == 0:
                print(f"  ⚠️  WARNING: No extractions were cached! Check if run_with_llm=True and extractions exist.")
        except Exception as e:
            # Non-critical - pipeline continues even if cache save fails
            import traceback
            print(f"  ⚠️  Cache save error (non-critical): {e}")
            traceback.print_exc()
    else:
        print("  ⚠️  LLM extractions not cached (run with --openai to enable caching)")
    
    # 5.5. Build segment vocabulary ONCE (before feature computation, for efficiency)
    # Try to load from cache first, then build if needed
    print("\n[5.5/10] Building segment vocabulary...")
    from features.segment_distribution import build_segment_vocabulary
    from features.segment_vocab_cache import load_cached_vocabulary, save_cached_vocabulary, clear_expired_cache
    
    # Clear expired cache entries (non-blocking)
    try:
        cleared = clear_expired_cache()
        if cleared > 0:
            print(f"  Cleared {cleared} expired cache entries")
    except Exception:
        pass
    
    all_companies_for_vocab = [target]  # Start with target
    for ticker, extracted in extracted_data.items():
        all_companies_for_vocab.append(extracted)
    
    # Try to load from cache (with error handling)
    segment_vocabulary = None
    try:
        cached_vocab = load_cached_vocabulary(target_id, all_companies_for_vocab)
        if cached_vocab:
            segment_vocabulary = cached_vocab
            print(f"  ✓ Loaded segment vocabulary from cache: {len(segment_vocabulary)} unique segments")
    except Exception as e:
        # If cache lookup fails, just build it
        print(f"  Warning: Cache lookup failed: {e}, building vocabulary")
        segment_vocabulary = None
    
    # Build vocabulary if not loaded from cache
    if segment_vocabulary is None:
        segment_vocabulary = build_segment_vocabulary(all_companies_for_vocab)
        print(f"  Built segment vocabulary: {len(segment_vocabulary)} unique segments")
        
        # Save to cache for future runs (non-blocking)
        try:
            save_cached_vocabulary(target_id, all_companies_for_vocab, segment_vocabulary)
            print(f"  ✓ Cached vocabulary for future runs")
        except Exception as e:
            # Non-critical - continue even if caching fails
            pass
    
    # 6. Feature computation
    print("\n[6/10] Computing features...")
    print(f"  Processing {len(shortlist_df)} shortlisted candidates...")
    feature_rows = []
    failed_companies = []
    
    for idx, candidate_row in shortlist_df.iterrows():
        ticker = candidate_row['ticker']
        pack = evidence_packs.get(ticker, {})
        extracted = extracted_data.get(ticker, {})
        
        # Debug: Check if we have evidence and extraction
        if not pack:
            print(f"    WARNING: {ticker} has no evidence pack")
        if not extracted:
            print(f"    WARNING: {ticker} has no LLM extraction data")
        
        try:
            # Pass segment vocabulary to compute_features for S computation (now precomputed!)
            candidate_row['segment_vocabulary'] = segment_vocabulary
            features = compute_features(target, candidate_row, extracted, pack, run_with_openai=args.openai)
            
            # Convert concept_matches to JSON string for CSV
            concept_matches_json = json.dumps(features.get('concept_matches', []))
            initiatives_json = json.dumps(features.get('initiatives', []))
            # Convert segment mix dicts to JSON strings for CSV
            segment_mix_target_json = json.dumps(features.get('segment_mix_target', {}))
            segment_mix_candidate_json = json.dumps(features.get('segment_mix_candidate', {}))
            
            feature_row = {
                'ticker': ticker,
                'name': candidate_row.get('name', ''),
                'exchange': candidate_row.get('exchange', ''),
                'P': features['P'],
                'C': features['C'],
                'S': features['S'],  # Segment distribution similarity (M removed - redundant)
                'B': features.get('B', 0.0),  # Business model similarity
                'V': features.get('V', 0.0),  # Vertical similarity (multi-hot encoding)
                'I': features['I'],
                'E': features['E'],
                'R': features['R'],
                'same_sic': features.get('same_sic', 0.0),  # SIC industry bonus
                # Precomputed score (reuse to avoid redundant computation)
                'score_linear': features.get('score_linear', 0.0),
                # Segment distribution S details (for explainability and gate reuse)
                'sim_cosine': features.get('sim_cosine', 0.0),
                'entropy_target': features.get('entropy_target', 0.0),
                'entropy_candidate': features.get('entropy_candidate', 0.0),
                'penalty_entropy': features.get('penalty_entropy', 0.0),
                'segment_mix_target': segment_mix_target_json,
                'segment_mix_candidate': segment_mix_candidate_json,
                'product_hits': features['product_hits'],
                'customer_hits': features['customer_hits'],
                'LLM_confidence': features['LLM_confidence'],
                'confidence_final': features['confidence_final'],
                'concept_matches': concept_matches_json,
                'initiatives': initiatives_json,
                # Store seg_s_result for gate reuse (avoid recomputing)
                '_seg_s_result': features.get('_seg_s_result'),
                'business_activity': ', '.join(extracted.get('business_activity', [])),
                'customer_segment': ', '.join(extracted.get('customer_segment', [])),
                'segment_mix': json.dumps(extracted.get('segment_mix', {})),
                'evidence_urls': '; '.join([s.get('url', '') for s in pack.get('sources', [])]),
                'evidence_quotes': extracted.get('evidence', [{}])[0].get('quote', '') if extracted.get('evidence') else '',
                'prompt_version': prompt_version,
                # Business model fields
                'business_model_type': extracted.get('business_model_type', 'other'),
                'services_share_estimate': extracted.get('services_share_estimate', 0.5),
                'revenue_model': ', '.join(extracted.get('revenue_model', ['other'])),
                'has_professional_services': extracted.get('has_professional_services', False),
                'has_managed_services': extracted.get('has_managed_services', False),
                'has_software_product': extracted.get('has_software_product', False)
            }
            feature_rows.append(feature_row)
        except Exception as e:
            # Log the error but continue processing other companies
            failed_companies.append({
                'ticker': ticker,
                'name': candidate_row.get('name', 'Unknown'),
                'error': str(e)
            })
            import traceback
            print(f"    ERROR: Failed to compute features for {ticker} ({candidate_row.get('name', 'Unknown')}): {e}")
            if len(failed_companies) <= 5:  # Only show full traceback for first 5 errors
                traceback.print_exc()
            continue
    
    if failed_companies:
        print(f"\n  ⚠️  Failed to compute features for {len(failed_companies)} companies:")
        for fc in failed_companies[:10]:  # Show first 10 failures
            print(f"    - {fc['ticker']} ({fc['name']}): {fc['error']}")
        if len(failed_companies) > 10:
            print(f"    ... and {len(failed_companies) - 10} more")
    
    features_df = pd.DataFrame(feature_rows)
    print(f"✓ Computed features for {len(features_df)} candidates (expected {len(shortlist_df)})")
    
    if len(features_df) < len(shortlist_df):
        print(f"  ⚠️  WARNING: Only {len(features_df)}/{len(shortlist_df)} companies have features computed!")
        print(f"  Missing companies: {len(shortlist_df) - len(features_df)}")
    
    # 6.5. Fit explainer model + compute SHAP values
    print("\n[7/10] Fitting explainer model + computing SHAP values...")
    
    # Import SCORING_CONFIG (needed for SHAP computation)
    from scoring_config import SCORING_CONFIG
    
    # score_linear already computed in compute_features() - reuse it (no redundant computation)
    # If missing (shouldn't happen), compute it
    if 'score_linear' not in features_df.columns:
        from ranker.scorer_rule import compute_base_score
        score_linear_list = []
        for _, row in features_df.iterrows():
            features_dict = {
                'P': row['P'], 'C': row['C'], 'S': row['S'],  # M removed
                'I': row.get('I', 0.5), 'E': row.get('E', 0.0), 'R': row.get('R', 0.5),
                'same_sic': row.get('same_sic', 0.0)
            }
            score_linear = compute_base_score(features_dict, SCORING_CONFIG)
            score_linear_list.append(score_linear)
        features_df['score_linear'] = score_linear_list
    
    # Also compute just P, C, S for SHAP (legacy - SHAP uses 3 features now, M removed)
    feature_weights = SCORING_CONFIG.get('weights', {})
    w_p = feature_weights.get('P', 0.35)
    w_c = feature_weights.get('C', 0.30)
    w_s = feature_weights.get('S', 0.35)  # M weight redistributed to S
    
    # Compute P, C, S only linear score for SHAP (legacy compatibility, M removed)
    features_df['score_linear_pcs'] = (
        w_p * features_df['P'] +
        w_c * features_df['C'] +
        w_s * features_df['S']
    )
    
    # Try to compute SHAP values (optional - for explainability)
    # Note: Environment variables are set at the top of the file to prevent segfaults
    if args.skip_shap:
        print("  Skipping SHAP computation (--skip-shap flag set)")
        # Add placeholder columns with NaN
        for col in ['P', 'C', 'S']:
            features_df[f'shap_{col}'] = np.nan
        features_df['shap_base_value'] = np.nan
        features_df['score_model'] = np.nan
    else:
        try:
            import xgboost as xgb
            import shap
            from features.shap_cache import load_cached_shap, save_cached_shap, clear_expired_shap_cache
            
            # Clear expired cache entries (non-blocking)
            try:
                cleared = clear_expired_shap_cache()
                if cleared > 0:
                    print(f"  Cleared {cleared} expired SHAP cache entries")
            except Exception:
                pass
            
            # Prepare feature matrix and target (M removed, now 3 features)
            feature_cols = ['P', 'C', 'S']
            
            # Validate data: check for NaN, inf, or missing columns
            for col in feature_cols:
                if col not in features_df.columns:
                    raise ValueError(f"Missing feature column: {col}")
                if features_df[col].isna().any():
                    print(f"  Warning: Found NaN values in {col}, filling with 0")
                    features_df[col] = features_df[col].fillna(0.0)
                if np.isinf(features_df[col]).any():
                    print(f"  Warning: Found inf values in {col}, replacing with 0")
                    features_df.loc[np.isinf(features_df[col]), col] = 0.0
            
            # Ensure numeric types
            X = features_df[feature_cols].values.astype(np.float64)
            # Use P,C,S only linear score for SHAP (for consistency with 3-feature model, M removed)
            y = features_df['score_linear_pcs'].values.astype(np.float64)
            
            # Validate shapes
            if len(X) == 0:
                raise ValueError("Empty feature matrix")
            if len(y) == 0:
                raise ValueError("Empty target vector")
            if X.shape[1] != 3:
                raise ValueError(f"Expected 3 features (P, C, S), got {X.shape[1]}")
            
            # Try to load from cache first
            cached_shap = load_cached_shap(target_id, features_df, feature_cols)
            
            if cached_shap is not None:
                # Use cached model and SHAP values
                model = cached_shap['model']
                shap_values = cached_shap['shap_values']
                base_value = cached_shap['base_value']
                features_df['score_model'] = model.predict(X)
                
                print(f"  ✓ Loaded SHAP values from cache (saved {cached_shap['metadata'].get('created_at', 'unknown')})")
            else:
                # Train small XGBoost model (pseudo-label approximation)
                model = xgb.XGBRegressor(
                    max_depth=2,
                    n_estimators=30,
                    learning_rate=0.05,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    objective='reg:squarederror',
                    random_state=42,
                    tree_method='hist',  # More stable than default on macOS
                )
                
                model.fit(X, y)
                features_df['score_model'] = model.predict(X)
                
                # Compute SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)  # shape: (n_samples, 3) - M removed
                
                # Handle expected_value - it can be scalar or array
                expected_value = explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    base_value = float(expected_value[0] if expected_value.size > 0 else expected_value.item())
                else:
                    base_value = float(expected_value)
                
                # Save to cache for future runs
                try:
                    save_cached_shap(target_id, features_df, model, shap_values, base_value, feature_cols)
                    print(f"  ✓ Cached SHAP model and values for future runs")
                except Exception as e:
                    # Non-critical - continue even if caching fails
                    pass
            
            # Add SHAP columns to dataframe
            for j, col in enumerate(feature_cols):
                features_df[f'shap_{col}'] = shap_values[:, j]
            
            features_df['shap_base_value'] = base_value
            
            print(f"✓ Computed SHAP values for {len(features_df)} candidates")
            print(f"  Model score correlation with linear: {np.corrcoef(features_df['score_linear_pcs'], features_df['score_model'])[0,1]:.3f}")
            
        except ImportError as e:
            print(f"  Warning: SHAP/XGBoost not available ({e}). Skipping SHAP computation.")
            # Add placeholder columns with NaN (score_linear already computed above, M removed)
            for col in ['P', 'C', 'S']:
                features_df[f'shap_{col}'] = np.nan
            features_df['shap_base_value'] = np.nan
            features_df['score_model'] = np.nan
        except Exception as e:
            import traceback
            error_msg = str(e)
            # Print full traceback for debugging
            print(f"  Warning: SHAP computation failed. Error: {error_msg}")
            print(f"  Full traceback:")
            traceback.print_exc()
            
            # Check if it's an OpenMP error
            if 'libomp' in error_msg.lower() or 'openmp' in error_msg.lower() or 'omp' in error_msg.lower():
                print(f"  Retrying with KMP_DUPLICATE_LIB_OK=TRUE...")
                # Set environment variable and retry (os already imported at top)
                os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                os.environ['NUMBA_DISABLE_JIT'] = '1'
                try:
                    # Retry SHAP computation
                    import xgboost as xgb
                    import shap
                    
                    # Prepare feature matrix and target (with validation, M removed)
                    feature_cols = ['P', 'C', 'S']
                    
                    # Validate and clean data
                    for col in feature_cols:
                        if features_df[col].isna().any():
                            features_df[col] = features_df[col].fillna(0.0)
                        if np.isinf(features_df[col]).any():
                            features_df.loc[np.isinf(features_df[col]), col] = 0.0
                    
                    X = features_df[feature_cols].values.astype(np.float64)
                    y = features_df['score_linear_pcs'].values.astype(np.float64)
                    
                    # Train small XGBoost model (pseudo-label approximation)
                    model = xgb.XGBRegressor(
                        max_depth=2,
                        n_estimators=30,
                        learning_rate=0.05,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        objective='reg:squarederror',
                        random_state=42,
                        tree_method='hist',  # More stable than default on macOS
                    )
                    
                    model.fit(X, y)
                    features_df['score_model'] = model.predict(X)
                    
                    # Compute SHAP values
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)  # shape: (n_samples, 3) - M removed
                    
                    # Handle expected_value - it can be scalar or array
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, np.ndarray):
                        base_value = float(expected_value[0] if expected_value.size > 0 else expected_value.item())
                    else:
                        base_value = float(expected_value)
                    
                    # Add SHAP columns to dataframe
                    for j, col in enumerate(feature_cols):
                        features_df[f'shap_{col}'] = shap_values[:, j]
                    
                    features_df['shap_base_value'] = base_value
                    
                    print(f"✓ Computed SHAP values for {len(features_df)} candidates (after retry)")
                    print(f"  Model score correlation with linear: {np.corrcoef(features_df['score_linear_pcms'], features_df['score_model'])[0,1]:.3f}")
                except Exception as e2:
                    print(f"  Warning: SHAP computation failed after retry ({e2}). Continuing without SHAP.")
                    # Add placeholder columns with NaN (score_linear already computed above)
                    for col in ['P', 'C', 'M', 'S']:
                        features_df[f'shap_{col}'] = np.nan
                    features_df['shap_base_value'] = np.nan
                    features_df['score_model'] = np.nan
            else:
                print(f"  Continuing without SHAP values (score_linear is still computed for ranking).")
                # Add placeholder columns with NaN (score_linear already computed above)
                for col in ['P', 'C', 'M', 'S']:
                    features_df[f'shap_{col}'] = np.nan
                features_df['shap_base_value'] = np.nan
                features_df['score_model'] = np.nan
    
    # 8. KNN leaderboard (semantic similarity only)
    print("\n[8/10] Creating KNN leaderboard...")
    knn_df = candidates_df[['ticker', 'name', 'exchange', 'S_fast', 'P_kw', 'C_kw']].copy()
    knn_df = knn_df.rename(columns={'S_fast': 'knn_score'})
    knn_df = knn_df.sort_values('knn_score', ascending=False)
    knn_df['rank_knn'] = range(1, len(knn_df) + 1)
    
    knn_path = os.path.join(OUTPUTS_DIR, f'{target_id}_knn.csv')
    try:
        export_leaderboard(knn_df, knn_path, 'knn')
        if not os.path.exists(knn_path):
            raise FileNotFoundError(f"KNN CSV was not created at {knn_path}")
    except Exception as e:
        print(f"  ERROR: Failed to export KNN leaderboard: {e}")
        # Try direct CSV save as fallback
        try:
            knn_df.to_csv(knn_path, index=False)
            print(f"  ✓ Saved KNN leaderboard via fallback method: {knn_path}")
        except Exception as e2:
            print(f"  CRITICAL: Could not save KNN leaderboard: {e2}")
    
    # 9. Rule-based ranking and gates
    print("\n[9/10] Computing rule-based scores and applying gates...")
    scored_rows = []
    from ranker.scorer_rule import (
        rule_score, compute_base_score, apply_business_model_penalty,
        compute_segment_overlap, gate_segments
    )
    from scoring_config import SCORING_CONFIG
    
    # Precompute model similarities for all candidates (for efficiency)
    # NOTE: Segment vocabulary already built earlier, segment similarity already in S feature
    from features.model_similarity import compute_model_similarities, convert_legacy_revenue_model, convert_legacy_delivery_modes
    
    # Ensure target has 3-layer model (convert from legacy if needed)
    from features.model_similarity import infer_archetypes_from_legacy
    
    if not target.get('revenue_archetypes'):
        target['revenue_archetypes'] = infer_archetypes_from_legacy(
            target.get('business_model_type'),
            target.get('revenue_model', []),
            target.get('services_share_estimate', 0.5)
        )
    if not target.get('revenue_channels') and target.get('revenue_model'):
        target['revenue_channels'] = convert_legacy_revenue_model(target.get('revenue_model', []))
    if not target.get('revenue_model_mix') and target.get('revenue_channels'):
        target['revenue_model_mix'] = target['revenue_channels'].copy()  # Legacy compatibility
    if not target.get('delivery_modes'):
        target['delivery_modes'] = convert_legacy_delivery_modes(
            target.get('business_model_type'),
            target.get('has_software_product', False),
            target.get('has_professional_services', False),
            target.get('has_managed_services', False)
        )
    
    # Extract segment similarity details from S feature computation (already done in compute_features)
    # No need to recompute - just extract from existing S computation
    from features.segment_distribution import compute_segment_similarity
    
    for idx, feature_row in features_df.iterrows():
        ticker = feature_row['ticker']
        extracted = extracted_data.get(ticker, {})
        
        # Segment similarity already computed in S feature - extract it if available
        # Otherwise compute it once (for gating, not for scoring)
        if 'segment_similarity' not in features_df.columns or pd.isna(features_df.at[idx, 'segment_similarity']):
            try:
                seg_sim_result = compute_segment_similarity(
                    target,
                    extracted,
                    vocabulary=segment_vocabulary
                )
                features_df.at[idx, 'segment_similarity'] = seg_sim_result.get('segment_similarity', 0.0)
                features_df.at[idx, 'concentration_penalty'] = seg_sim_result.get('concentration_penalty', 1.0)
                features_df.at[idx, 'normalized_entropy'] = seg_sim_result.get('normalized_entropy', 0.0)
            except Exception as e:
                # Fallback to legacy binary overlap if vector-based approach fails
                row_for_segments = {
                    'customer_segment': extracted.get('customer_segment', []),
                    'C': feature_row['C']
                }
                segments_overlap = compute_segment_overlap(row_for_segments, target)
                features_df.at[idx, 'segments_overlap'] = segments_overlap
                features_df.at[idx, 'segment_similarity'] = 0.0
                features_df.at[idx, 'concentration_penalty'] = 1.0
                features_df.at[idx, 'normalized_entropy'] = 0.0
        
        # Ensure candidate has 3-layer model (convert from legacy if needed)
        if not extracted.get('revenue_archetypes'):
            extracted['revenue_archetypes'] = infer_archetypes_from_legacy(
                extracted.get('business_model_type'),
                extracted.get('revenue_model', []),
                extracted.get('services_share_estimate', 0.5)
            )
        if not extracted.get('revenue_channels') and extracted.get('revenue_model'):
            extracted['revenue_channels'] = convert_legacy_revenue_model(extracted.get('revenue_model', []))
        if not extracted.get('revenue_model_mix') and extracted.get('revenue_channels'):
            extracted['revenue_model_mix'] = extracted['revenue_channels'].copy()  # Legacy compatibility
        if not extracted.get('delivery_modes'):
            extracted['delivery_modes'] = convert_legacy_delivery_modes(
                extracted.get('business_model_type'),
                extracted.get('has_software_product', False),
                extracted.get('has_professional_services', False),
                extracted.get('has_managed_services', False)
            )
        
        # Compute model similarities (3-layer model)
        try:
            similarities = compute_model_similarities(extracted, target)
            features_df.at[idx, 'archetype_similarity'] = similarities.get('archetype_similarity', 0.0)
            features_df.at[idx, 'channel_similarity'] = similarities.get('channel_similarity', 0.0)
            features_df.at[idx, 'delivery_mode_similarity'] = similarities.get('delivery_mode_similarity', 0.0)
            features_df.at[idx, 'revenue_model_similarity'] = similarities.get('revenue_model_similarity', 0.0)  # Backward compat
        except Exception as e:
            # If similarity computation fails, set to 0.0
            features_df.at[idx, 'archetype_similarity'] = 0.0
            features_df.at[idx, 'channel_similarity'] = 0.0
            features_df.at[idx, 'delivery_mode_similarity'] = 0.0
            features_df.at[idx, 'revenue_model_similarity'] = 0.0
    
    # Single consolidated loop: compute scores and apply gates
    for _, feature_row in features_df.iterrows():
        ticker = feature_row['ticker']
        extracted = extracted_data.get(ticker, {})
        
        features_dict = {
            'P': feature_row['P'],
            'C': feature_row['C'],
            'S': feature_row['S'],  # M removed - redundant with S
            'B': feature_row.get('B', 0.0),  # NEW: Business model similarity
            'I': feature_row['I'],
            'E': feature_row['E'],
            'R': feature_row['R'],
            'product_hits': int(feature_row.get('product_hits', 0)),
            'customer_hits': int(feature_row.get('customer_hits', 0)),
            'segments_overlap': int(feature_row.get('segments_overlap', 0)),  # Legacy (for backward compat)
            'segment_similarity': float(feature_row.get('segment_similarity', 0.0)),  # New: continuous vector similarity
            'concentration_penalty': float(feature_row.get('concentration_penalty', 1.0)),  # New: entropy-based penalty
            'normalized_entropy': float(feature_row.get('normalized_entropy', 0.0)),  # New: entropy of distribution
            'archetype_similarity': float(feature_row.get('archetype_similarity', 0.0)),
            'channel_similarity': float(feature_row.get('channel_similarity', 0.0)),
            'delivery_mode_similarity': float(feature_row.get('delivery_mode_similarity', 0.0)),
            'revenue_model_similarity': float(feature_row.get('revenue_model_similarity', 0.0)),  # Backward compat
            'same_sic': float(feature_row.get('same_sic', 0.0))
        }
        
        # Compute scores using configurable functions
        # 1. Base linear score (reuse precomputed if available, avoid redundant computation)
        score_linear = feature_row.get('score_linear')
        if score_linear is None or pd.isna(score_linear):
            score_linear = compute_base_score(features_dict, SCORING_CONFIG)
        
        # 1.5. Compute archetype similarity (NEW: economic signature matching)
        archetype_similarity = 0.5  # Default neutral
        archetype_info = {}
        
        # Get target's economic_signature - extract on-the-fly if not in target.json
        target_sig = target.get('extracted_data', {}).get('economic_signature', {}) or target.get('economic_signature', {})
        if not target_sig:
            # Extract economic_signature from target's existing fields on-the-fly
            # Use the same extraction logic as candidates
            try:
                from features.economic_signature import extract_economic_signature_from_llm
                # Use target as if it were extracted_data - it has the same fields
                target_sig = extract_economic_signature_from_llm(target)
            except Exception:
                # If extraction fails, target_sig stays empty (will use neutral score)
                target_sig = {}
        
        # CRITICAL: If target_sig is missing NEW archetype fields, infer them from business description
        # This is needed because targets created before the NEW schema don't have these fields
        if not target_sig or not target_sig.get('capacity_unit') or target_sig.get('capacity_unit') == 'none':
            try:
                from features.archetype_inference import infer_archetype_fields_from_target
                inferred_fields = infer_archetype_fields_from_target(target)
                # Initialize target_sig if empty
                if not target_sig:
                    target_sig = {}
                # Merge inferred fields into target_sig (don't overwrite if LLM provided them)
                for key, value in inferred_fields.items():
                    if key not in target_sig or not target_sig.get(key) or target_sig.get(key) == 'none' or target_sig.get(key) == []:
                        target_sig[key] = value
            except Exception as e:
                # If inference fails, continue with what we have
                import warnings
                warnings.warn(f"Archetype inference failed for target: {e}", stacklevel=2)
                pass
        
        candidate_sig = extracted.get('economic_signature', {})
        
        if target_sig and candidate_sig:
            try:
                from features.archetypes import load_archetypes, pair_archetype_similarity
                archetypes = load_archetypes()
                archetype_info = pair_archetype_similarity(
                    target_sig_dict=target_sig,
                    candidate_sig_dict=candidate_sig,
                    archetypes=archetypes
                )
                archetype_similarity = archetype_info.get('similarity', 0.5)
            except Exception as e:
                # If archetype computation fails, continue with neutral score
                import warnings
                warnings.warn(f"Archetype similarity computation failed for {ticker}: {e}", stacklevel=2)
        
        # Mix archetype similarity into base score (configurable weight)
        w_arch = SCORING_CONFIG.get('archetype', {}).get('weight', 0.2)  # Default 20% weight
        score_with_archetype = (1.0 - w_arch) * score_linear + w_arch * archetype_similarity
        score_linear = float(np.clip(score_with_archetype, 0.0, 1.0))
        
        # Store archetype info for later use
        features_dict['archetype_similarity_new'] = archetype_similarity
        features_dict['archetype_info'] = archetype_info
        
        # 2. Apply business model penalty (uses target profile for dynamic anchor)
        penalty_value, score_adjusted = apply_business_model_penalty(
            score_linear, extracted, SCORING_CONFIG, target
        )
        
        # 3. Full rule score (includes all gates)
        # Pass vocabulary and precomputed S result to rule_score to avoid recomputing
        features_dict['segment_vocabulary'] = segment_vocabulary
        features_dict['_seg_s_result'] = feature_row.get('_seg_s_result')  # Reuse precomputed segment similarity
        score_100, pct_dict, passed_gates, gate_details, score_adjusted_final = rule_score(
            features_dict,
            target_profile=target,
            extracted_data=extracted,
            config=SCORING_CONFIG
        )
        
        # Use score_adjusted from rule_score (includes all penalties: discipline + economic + business model)
        # This ensures ranking uses the correct final score
        score_adjusted = score_adjusted_final
        
        # For testing, include all candidates even if gates fail (but mark it)
        scored_row = {
            **feature_row,
            'ml_score': score_100,  # Final score on 0-100 scale
            'score_linear': score_linear,  # Base linear score (0-1 scale)
            'score_adjusted': score_adjusted,  # After all penalties (0-1 scale) - from rule_score
            **pct_dict,
            'passed_gates': passed_gates,
            'gate_business_model': gate_details.get('business_model', False),
            'gate_segments': gate_details.get('segments', False),
            'gate_hospitality_keywords': gate_details.get('hospitality_keywords', True),  # NEW: Hospitality keywords gate
            'gate_economic_engine': gate_details.get('economic_engine', True),  # NEW: Economic engine gate
            'gate_product_hits': gate_details.get('product_hits', False),
            'gate_customer_hits': gate_details.get('customer_hits', False),
            'business_model_type': extracted.get('business_model_type', 'other'),
            'services_share_estimate': extracted.get('services_share_estimate', 0.5),
            'penalty_producty': penalty_value,  # Penalty value (0-1 scale)
            'segments_overlap': int(feature_row.get('segments_overlap', 0))
        }
        # Include SHAP columns if available (from step 7)
        if 'shap_P' in feature_row:
            scored_row['score_model'] = feature_row.get('score_model', np.nan)
            scored_row['shap_P'] = feature_row.get('shap_P', np.nan)
            scored_row['shap_C'] = feature_row.get('shap_C', np.nan)
            scored_row['shap_M'] = feature_row.get('shap_M', np.nan)
            scored_row['shap_S'] = feature_row.get('shap_S', np.nan)
            scored_row['shap_base_value'] = feature_row.get('shap_base_value', np.nan)
        scored_rows.append(scored_row)
    
    ranked_df = pd.DataFrame(scored_rows)
    if len(ranked_df) > 0:
        # Exclude the target itself (if ticker is in target.json)
        target_ticker_raw = target.get('ticker') or target.get('ticker_symbol') or ''
        target_ticker = str(target_ticker_raw).upper() if target_ticker_raw else ''
        if target_ticker:
            before_count = len(ranked_df)
            ranked_df = ranked_df[ranked_df['ticker'].str.upper() != target_ticker].copy()
            if len(ranked_df) < before_count:
                print(f"  Excluded target company ({target_ticker}) from results")
        
        # ============================================================
        # CRITICAL: SORT BY score_adjusted FIRST (includes all penalties!)
        # ============================================================
        # Determine sort column (score_adjusted includes all penalties: discipline + economic + business model)
        sort_col = 'score_adjusted' if 'score_adjusted' in ranked_df.columns else 'ml_score'
        if sort_col not in ranked_df.columns:
            sort_col = 'score_linear' if 'score_linear' in ranked_df.columns else 'score_linear_pcms'
        
        # Check if score_adjusted is all zeros (bug fix: fallback to score_linear if score_adjusted is broken)
        if sort_col == 'score_adjusted' and (ranked_df[sort_col].max() == 0.0 or ranked_df[sort_col].isna().all()):
            print(f"  ⚠️  Warning: {sort_col} is all zeros or NaN, falling back to score_linear")
            sort_col = 'score_linear' if 'score_linear' in ranked_df.columns else 'score_linear_pcms'
        
        # Sort by sort_col (descending - highest scores first)
        ranked_df = ranked_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        print(f"  ✓ Sorted {len(ranked_df)} candidates by {sort_col} (includes all penalties)")
        
        # ============================================================
        # CRITICAL: APPLY GATES BEFORE RANKING (not after!)
        # ============================================================
        print(f"\n  Applying gates to {len(ranked_df)} candidates...")
        
        # Step 1: Filter to only companies that passed ALL gates
        passed_gates_df = ranked_df[ranked_df['passed_gates'] == True].copy()
        print(f"  Companies passing all gates: {len(passed_gates_df)}")
        
        # DEBUG: Log gate failures for top 20 candidates
        print(f"\n  [DEBUG] Gate failure analysis for top 20 candidates:")
        print(f"  {'='*100}")
        print(f"  {'Ticker':<8} {'P':<6} {'C':<6} {'S':<6} {'BM Type':<20} {'Svcs%':<6} {'Failed Gates':<30}")
        print(f"  {'-'*100}")
        
        for idx, row in ranked_df.head(20).iterrows():
            ticker = row.get('ticker', 'N/A')
            P = row.get('P', 0.0)
            C = row.get('C', 0.0)
            S = row.get('S', 0.0)
            sim_cosine = row.get('sim_cosine', 0.0)
            segment_similarity = sim_cosine if sim_cosine > 0 else S
            
            # Get extracted data for business model info
            ticker_val = row.get('ticker', '')
            extracted = extracted_data.get(ticker_val, {}) if ticker_val else {}
            business_model_type = extracted.get('business_model_type', 'unknown')
            services_share = extracted.get('services_share_estimate', 0.0)
            
            # Check which gates failed
            failed_gates = []
            # Business model gate removed - soft penalties handle filtering
            # gate_business_model is always True now (for reporting only)
            if not row.get('gate_segments', True):
                failed_gates.append('SEG')
            if not row.get('gate_product_hits', True):
                failed_gates.append('P_HITS')
            if not row.get('gate_customer_hits', True):
                failed_gates.append('C_HITS')
            
            failed_str = ', '.join(failed_gates) if failed_gates else 'PASS'
            
            print(f"  {ticker:<8} {P:<6.2f} {C:<6.2f} {segment_similarity:<6.2f} {business_model_type:<20} {services_share:<6.2f} {failed_str:<30}")
        
        print(f"  {'='*100}\n")
        
        # Step 2: If fewer than 10 passed, try progressive fallbacks to get at least 10
        if len(passed_gates_df) < 10:
            print(f"  WARNING: Only {len(passed_gates_df)} companies passed all gates (target: 10)")
            
            # Fallback 1: Business model gate removed - all companies are ranked (soft penalties handle filtering)
            # Since ranked_df is already sorted by score_adjusted, just take top 10
            print(f"  Trying fallback 1: using top 10 by score_adjusted (soft penalties already applied)...")
            fallback1_df = ranked_df.head(10).copy()
            print(f"  Using top 10 by {sort_col} (business model gate removed - soft penalties handle filtering)")
            
            if len(fallback1_df) >= 10:
                passed_gates_df = fallback1_df
                print(f"  ✓ Using fallback 1: {len(passed_gates_df)} companies that passed business_model gate")
            elif len(fallback1_df) >= 3:
                # If we have at least 3, use it but warn
                passed_gates_df = fallback1_df
                print(f"  ⚠️  Using fallback 1: {len(passed_gates_df)} companies (fewer than 10, but at least 3)")
            else:
                # Fallback 2: Just filter by score (no gates) - last resort
                print(f"  Trying fallback 2: filtering by score only (no gates)...")
                # Sort by score and take top candidates
                sort_col = 'score_adjusted' if 'score_adjusted' in ranked_df.columns else 'score_linear'
                if sort_col not in ranked_df.columns:
                    sort_col = 'ml_score' if 'ml_score' in ranked_df.columns else 'score_linear_pcms'
                
                fallback2_df = ranked_df.sort_values(sort_col, ascending=False).head(20).copy()
                print(f"  Companies in top 20 by score: {len(fallback2_df)}")
                
                if len(fallback2_df) >= 10:
                    passed_gates_df = fallback2_df.head(10)
                    print(f"  ⚠️  Using fallback 2: Top 10 by score (gates were too strict)")
                elif len(fallback2_df) >= 3:
                    passed_gates_df = fallback2_df
                    print(f"  ⚠️  Using fallback 2: Top {len(fallback2_df)} by score (gates were too strict)")
                else:
                    # Last resort: use whatever we have
                    if len(fallback1_df) > 0:
                        passed_gates_df = fallback1_df
                        print(f"  ⚠️  Using fallback 1: {len(passed_gates_df)} companies (gates were too strict)")
                    else:
                        print(f"  CRITICAL: No companies passed business_model gate!")
                        print(f"  This indicates a problem with target profile or gate thresholds.")
                        # Still try to show top 10 by score as absolute last resort
                        passed_gates_df = ranked_df.sort_values(sort_col, ascending=False).head(10).copy()
                        print(f"  ⚠️  Showing top 10 by score as absolute last resort (gates failed)")
        
        # Step 3: Now sort the filtered companies by score
        if len(passed_gates_df) > 0:
            sort_col = 'score_adjusted' if 'score_adjusted' in passed_gates_df.columns else 'ml_score'
            if sort_col not in passed_gates_df.columns:
                sort_col = 'score_linear' if 'score_linear' in passed_gates_df.columns else 'score_linear_pcms'
            
            passed_gates_df = passed_gates_df.sort_values(sort_col, ascending=False)
            passed_gates_df['rank_ml'] = range(1, len(passed_gates_df) + 1)
            
            # Update ranked_df to show filtered results (for analysis/debugging)
            # Keep original ranked_df for full analysis, but use passed_gates_df for final output
            ranked_df = passed_gates_df.copy()
            print(f"  ✓ Filtered to {len(ranked_df)} companies that passed gates (sorted by score)")
        else:
            # No companies passed gates - this is a problem
            print(f"  ERROR: No companies passed gates! Cannot generate valid comps.")
            print(f"  This suggests:")
            print(f"    1. Target profile may be too restrictive")
            print(f"    2. Gate thresholds may be too strict")
            print(f"    3. Candidate pool may not contain suitable matches")
            # Keep ranked_df empty or with a warning row
            ranked_df = pd.DataFrame()
            ranked_df['rank_ml'] = []
            ranked_df['ml_score'] = []
    else:
        ranked_df['rank_ml'] = []
        ranked_df['ml_score'] = []
    
    ranked_path = os.path.join(OUTPUTS_DIR, f'{target_id}_ranked.csv')
    try:
        export_leaderboard(ranked_df, ranked_path, 'ranked')
        if not os.path.exists(ranked_path):
            raise FileNotFoundError(f"Ranked CSV was not created at {ranked_path}")
        print(f"✓ Ranked {len(ranked_df)} candidates and saved to {ranked_path}")
    except Exception as e:
        print(f"  ERROR: Failed to export ranked leaderboard: {e}")
        # Try direct CSV save as fallback
        try:
            ranked_df.to_csv(ranked_path, index=False)
            print(f"  ✓ Saved ranked leaderboard via fallback method: {ranked_path}")
        except Exception as e2:
            print(f"  CRITICAL: Could not save ranked leaderboard: {e2}")
    
    # 9.5. Export final top 10 comparables CSV (clean format)
    print("\n[9.5/10] Exporting final top 10 comparables CSV...")
    
    # Load runtime config for quality filters
    try:
        import yaml
        runtime_config_path = os.path.join(CONFIG_DIR, 'runtime.yaml')
        if os.path.exists(runtime_config_path):
            with open(runtime_config_path, 'r') as f:
                runtime_config = yaml.safe_load(f) or {}
        else:
            runtime_config = {}
    except Exception:
        runtime_config = {}
    
    # Determine sort column (same as used for ranked_df)
    sort_col = 'score_adjusted' if 'score_adjusted' in ranked_df.columns else 'ml_score'
    if sort_col not in ranked_df.columns:
        sort_col = 'score_linear' if 'score_linear' in ranked_df.columns else 'score_linear_pcms'
    
    # SOFT DISCIPLINE FILTERING: No hard gates - companies are ranked by score (which includes discipline penalties)
    # Companies with low discipline similarity are heavily downweighted (quadratic penalty), not removed
    # So we just take the top companies by score - the discipline penalty already handled the filtering
    passed_gates_df = ranked_df.copy()  # All companies pass (soft penalties handle filtering)
    
    # Quality filters: Only major exchanges, minimum score threshold
    quality_filters = runtime_config.get('final_comps', {})
    min_score_threshold = quality_filters.get('min_score_threshold', 0.0)
    allowed_exchanges = quality_filters.get('allowed_exchanges', ['NYSE', 'NASDAQ', 'NYSE MKT', 'NYSE Arca'])
    
    def apply_quality_filters(df):
        """Apply quality filters to a dataframe."""
        filtered = df.copy()
        if allowed_exchanges:
            filtered = filtered[
                filtered['exchange'].isin(allowed_exchanges) | 
                filtered['exchange'].isna()
            ].copy()
        if min_score_threshold > 0:
            filtered = filtered[filtered[sort_col] >= min_score_threshold].copy()
        return filtered
    
    # Step 2: If we have fewer than 10, use progressive fallbacks
    if len(passed_gates_df) >= 10:
        # We have enough companies that passed all gates
        passed_gates_df = apply_quality_filters(passed_gates_df)
        passed_gates_df = passed_gates_df.sort_values(sort_col, ascending=False)
        top_10_df = passed_gates_df.head(10).copy()
        print(f"  ✓ Selected top 10 from {len(passed_gates_df)} companies that passed all gates")
    else:
        # Not enough companies passed all gates - use fallbacks
        print(f"  ⚠️  Only {len(passed_gates_df)} companies passed all gates (target: 10)")
        
        # Fallback 1: Business model gate removed - all companies are ranked (soft penalties handle filtering)
        # Since ranked_df is already sorted by score_adjusted, just take top 10
        fallback1_df = ranked_df.head(10).copy()
        fallback1_df = apply_quality_filters(fallback1_df)
        # No need to sort again - already sorted by score_adjusted
        
        if len(fallback1_df) >= 10:
            top_10_df = fallback1_df.head(10).copy()
            print(f"  ✓ Using fallback 1: Top 10 companies that passed business_model gate ({len(fallback1_df)} available)")
        elif len(fallback1_df) >= 5:
            top_10_df = fallback1_df.head(10).copy() if len(fallback1_df) >= 10 else fallback1_df.copy()
            print(f"  ⚠️  Using fallback 1: {len(top_10_df)} companies that passed business_model gate (fewer than 10)")
        else:
            # Fallback 2: Just top by score (no gates) - last resort
            print(f"  ⚠️  Fallback 1 insufficient ({len(fallback1_df)} companies), using fallback 2: top by score")
            fallback2_df = ranked_df.copy()
            fallback2_df = apply_quality_filters(fallback2_df)
            fallback2_df = fallback2_df.sort_values(sort_col, ascending=False)
            top_10_df = fallback2_df.head(10).copy()
            print(f"  ⚠️  Using fallback 2: Top {len(top_10_df)} companies by score (gates were too strict)")
    
    # Load universe data to get website URLs
    universe_path = os.path.join(DATA_DIR, 'universe_us.csv')
    universe_df = None
    if os.path.exists(universe_path):
        try:
            universe_df = pd.read_csv(universe_path)
        except Exception as e:
            print(f"  Warning: Could not load universe.csv: {e}")
    
    # Build final comps dataframe with required fields
    if len(top_10_df) == 0:
        print(f"  WARNING: No companies to include in final comps (all failed gates)")
        final_comps_df = pd.DataFrame(columns=['name', 'url', 'exchange', 'ticker', 'business_activity', 'customer_segment', 'sic_industry'])
    else:
        final_comps_rows = []
        for _, row in top_10_df.iterrows():
            ticker = row['ticker']
            
            # Get website URL from universe or shortlist
            url = ''
            if universe_df is not None:
                universe_match = universe_df[universe_df['ticker'] == ticker]
                if not universe_match.empty:
                    url = universe_match.iloc[0].get('website', '')
            if not url:
                # Fallback to shortlist_df
                shortlist_match = shortlist_df[shortlist_df['ticker'] == ticker]
                if not shortlist_match.empty:
                    url = shortlist_match.iloc[0].get('website', '')
            
            # Get SIC industry (industry field from universe)
            sic_industry = ''
            if universe_df is not None:
                universe_match = universe_df[universe_df['ticker'] == ticker]
                if not universe_match.empty:
                    sic_industry = universe_match.iloc[0].get('industry', '')
            if not sic_industry:
                # Fallback to shortlist_df
                shortlist_match = shortlist_df[shortlist_df['ticker'] == ticker]
                if not shortlist_match.empty:
                    sic_industry = shortlist_match.iloc[0].get('industry', '')
            
            # Get business_activity from row (already a joined string, or from extracted_data)
            business_activity = row.get('business_activity', '')
            if not business_activity or business_activity == '':
                # Try to get from extracted_data
                extracted = extracted_data.get(ticker, {})
                activity_list = extracted.get('business_activity', [])
                if isinstance(activity_list, list) and len(activity_list) > 0:
                    business_activity = ', '.join(activity_list)
                elif activity_list:
                    business_activity = str(activity_list)
                # If still empty, try to infer from summary in universe data
                if not business_activity or business_activity == '':
                    if universe_df is not None:
                        universe_match = universe_df[universe_df['ticker'] == ticker]
                        if not universe_match.empty:
                            summary = universe_match.iloc[0].get('summary', '')
                            if summary and len(summary) > 0:
                                # Use first 200 chars of summary as business activity fallback
                                business_activity = str(summary)[:200].strip()
            
            # Get customer_segment from row (already a joined string, or from extracted_data)
            customer_segment = row.get('customer_segment', '')
            if not customer_segment or customer_segment == '':
                # Try to get from extracted_data
                extracted = extracted_data.get(ticker, {})
                segment_list = extracted.get('customer_segment', [])
                if isinstance(segment_list, list) and len(segment_list) > 0:
                    customer_segment = ', '.join(segment_list)
                elif segment_list:
                    customer_segment = str(segment_list)
            
            final_comps_rows.append({
                'name': row.get('name', ''),
                'url': url,
                'exchange': row.get('exchange', ''),
                'ticker': ticker,
                'business_activity': business_activity,
                'customer_segment': customer_segment,
                'sic_industry': sic_industry
            })
        
        final_comps_df = pd.DataFrame(final_comps_rows)
    
    # Main final_comps.csv - contains top 10 statistical comparables
    final_comps_path = os.path.join(OUTPUTS_DIR, f'{target_id}_final_comps.csv')
    try:
        final_comps_df.to_csv(final_comps_path, index=False)
        if not os.path.exists(final_comps_path):
            raise FileNotFoundError(f"Final comps CSV was not created at {final_comps_path}")
        print(f"✓ Exported final top {len(final_comps_df)} comparables to: {final_comps_path}")
    except Exception as e:
        print(f"  ERROR: Failed to export final comparables: {e}")
        raise
    
    # Print top 10 comparables to console
    print("\n" + "="*80)
    print("TOP 10 COMPARABLES:")
    print("="*80)
    for idx, row in final_comps_df.iterrows():
        rank = idx + 1
        name = row.get('name', 'N/A')
        ticker = row.get('ticker', 'N/A')
        exchange = row.get('exchange', 'N/A')
        business_activity = row.get('business_activity', 'N/A')
        customer_segment = row.get('customer_segment', 'N/A')
        print(f"\n{rank}. {name} ({ticker}) - {exchange}")
        print(f"   Business Activity: {business_activity[:150]}{'...' if len(str(business_activity)) > 150 else ''}")
        print(f"   Customer Segment: {customer_segment[:150]}{'...' if len(str(customer_segment)) > 150 else ''}")
    print("="*80)
    
    # 10. Metadata JSONL and Run Summary
    print("\n[10/10] Generating metadata JSONL and run summary...")
    metadata_path = os.path.join(OUTPUTS_DIR, f'{target_id}_comps_meta.jsonl')
    metadata_records = []
    for _, ranked_row in ranked_df.iterrows():
        ticker = ranked_row['ticker']
        pack = evidence_packs.get(ticker, {})
        extracted = extracted_data.get(ticker, {})
        candidate_row = shortlist_df[shortlist_df['ticker'] == ticker].iloc[0] if ticker in shortlist_df['ticker'].values else None
        
        # Collect 2-3 best evidence snippets showing why this is a comp
        # Prioritize: product/customer quotes from LLM extraction, then website/10-K sources
        evidence_snippets = []
        
        # First, try to get evidence from LLM extraction (most relevant)
        llm_evidence = extracted.get('evidence', [])
        if isinstance(llm_evidence, dict):
            # Evidence is organized by category (business_activity, customer_segment, etc.)
            for category in ['business_activity', 'customer_segment', 'products']:
                category_quotes = llm_evidence.get(category, [])
                if isinstance(category_quotes, list):
                    for quote_obj in category_quotes[:2]:  # Max 2 per category
                        if isinstance(quote_obj, dict) and quote_obj.get('quote'):
                            evidence_snippets.append({
                                'quote': quote_obj.get('quote', '')[:500],  # Limit length
                                'source': quote_obj.get('source', 'llm_extraction'),
                                'source_url': quote_obj.get('source_url', ''),
                                'category': category,
                                'reason': f"Matches target's {category.replace('_', ' ')}"
                            })
        elif isinstance(llm_evidence, list):
            # Evidence is a flat list
            for quote_obj in llm_evidence[:3]:
                if isinstance(quote_obj, dict) and quote_obj.get('quote'):
                    evidence_snippets.append({
                        'quote': quote_obj.get('quote', '')[:500],
                        'source': quote_obj.get('source', quote_obj.get('source_url', 'llm_extraction')),
                        'source_url': quote_obj.get('source_url', ''),
                        'category': quote_obj.get('category', 'general'),
                        'reason': 'LLM-extracted evidence showing similarity'
                    })
        
        # If we don't have enough evidence snippets, supplement with raw sources
        if len(evidence_snippets) < 2:
            sources = pack.get('sources', [])
            for source in sources[:3]:
                text = source.get('text', '')
                if text and len(text) > 50:
                    # Skip if we already have this URL
                    source_url = source.get('url', '')
                    if not any(e.get('source_url') == source_url for e in evidence_snippets):
                        evidence_snippets.append({
                            'quote': text[:500],
                            'source': source.get('type', 'unknown'),
                            'source_url': source_url,
                            'category': 'raw_source',
                            'reason': f"From {source.get('type', 'source')} evidence"
                        })
                    if len(evidence_snippets) >= 3:
                        break
        
        # Limit to 3 best snippets
        evidence_snippets = evidence_snippets[:3]
        
        # Map evidence to specific features (P, C, M, S)
        # Get candidate_row for Path B evidence
        candidate_row_for_evidence = shortlist_df[shortlist_df['ticker'] == ticker].iloc[0] if ticker in shortlist_df['ticker'].values else None
        evidence_by_feature = _map_evidence_to_features(
            extracted=extracted,
            pack=pack,
            target=target,
            candidate_row=candidate_row_for_evidence
        )
        
        # Build SHAP-like explanation: feature breakdown with weighted contributions
        # Load weights from scoring_config (same as rule_score uses)
        from scoring_config import SCORING_CONFIG
        feature_weights = SCORING_CONFIG.get('weights', {})
        
        # Calculate weighted contribution for each feature
        p_value = float(ranked_row.get('P', 0.0))
        c_value = float(ranked_row.get('C', 0.0))
        m_value = float(ranked_row.get('M', 0.0))
        s_value = float(ranked_row.get('S', 0.0))
        i_value = float(ranked_row.get('I', 0.0))
        e_value = float(ranked_row.get('E', 0.0))
        r_value = float(ranked_row.get('R', 0.0))
        
        w_p = feature_weights.get('P', 0.28)
        w_c = feature_weights.get('C', 0.28)
        w_m = feature_weights.get('M', 0.18)
        w_s = feature_weights.get('S', 0.16)
        w_i = feature_weights.get('I', 0.06)
        w_e = feature_weights.get('E', 0.03)
        w_r = feature_weights.get('R', 0.01)
        
        # Weighted contributions (feature_value * weight)
        weighted_p = p_value * w_p
        weighted_c = c_value * w_c
        weighted_m = m_value * w_m
        weighted_s = s_value * w_s
        weighted_i = i_value * w_i
        weighted_e = e_value * w_e
        weighted_r = r_value * w_r
        
        total_weighted = weighted_p + weighted_c + weighted_m + weighted_s + weighted_i + weighted_e + weighted_r
        
        # Build natural language explanation with SHAP + evidence
        score_linear = float(ranked_row.get('score_linear', 0.0)) if not pd.isna(ranked_row.get('score_linear', np.nan)) else float(ranked_row.get('ml_score', 0.0))
        natural_language_explanation = _build_natural_language_explanation(
            ranked_row=ranked_row,
            evidence_by_feature=evidence_by_feature,
            score_linear=score_linear
        )
        
        # Business model information
        business_model_info = {
            'business_model_type': extracted.get('business_model_type', 'unknown'),
            'services_share_estimate': float(extracted.get('services_share_estimate', 0.5) or 0.5),
            'revenue_model': extracted.get('revenue_model', []),
            'has_professional_services': extracted.get('has_professional_services', False),
            'has_managed_services': extracted.get('has_managed_services', False),
            'has_software_product': extracted.get('has_software_product', False),
            'gate_business_model': bool(ranked_row.get('gate_business_model', False)),
            'penalty_producty': float(ranked_row.get('penalty_producty', 0.0) or 0.0)
        }
        
        # Build explanation breakdown (structured)
        explanation = {
            'natural_language': natural_language_explanation,  # Human-readable explanation with evidence
            'feature_scores': {
                'P': {'raw': p_value, 'weight': w_p, 'weighted_contribution': weighted_p, 'pct_of_total': (weighted_p / total_weighted * 100) if total_weighted > 0 else 0.0},
                'C': {'raw': c_value, 'weight': w_c, 'weighted_contribution': weighted_c, 'pct_of_total': (weighted_c / total_weighted * 100) if total_weighted > 0 else 0.0},
                'M': {'raw': m_value, 'weight': w_m, 'weighted_contribution': weighted_m, 'pct_of_total': (weighted_m / total_weighted * 100) if total_weighted > 0 else 0.0},
                'S': {'raw': s_value, 'weight': w_s, 'weighted_contribution': weighted_s, 'pct_of_total': (weighted_s / total_weighted * 100) if total_weighted > 0 else 0.0},
                'I': {'raw': i_value, 'weight': w_i, 'weighted_contribution': weighted_i, 'pct_of_total': (weighted_i / total_weighted * 100) if total_weighted > 0 else 0.0},
                'E': {'raw': e_value, 'weight': w_e, 'weighted_contribution': weighted_e, 'pct_of_total': (weighted_e / total_weighted * 100) if total_weighted > 0 else 0.0},
                'R': {'raw': r_value, 'weight': w_r, 'weighted_contribution': weighted_r, 'pct_of_total': (weighted_r / total_weighted * 100) if total_weighted > 0 else 0.0}
            },
            'total_weighted_score': total_weighted,
            'top_contributors': sorted([
                {'feature': 'P', 'contribution': weighted_p, 'pct': (weighted_p / total_weighted * 100) if total_weighted > 0 else 0.0},
                {'feature': 'C', 'contribution': weighted_c, 'pct': (weighted_c / total_weighted * 100) if total_weighted > 0 else 0.0},
                {'feature': 'M', 'contribution': weighted_m, 'pct': (weighted_m / total_weighted * 100) if total_weighted > 0 else 0.0},
                {'feature': 'S', 'contribution': weighted_s, 'pct': (weighted_s / total_weighted * 100) if total_weighted > 0 else 0.0},
                {'feature': 'I', 'contribution': weighted_i, 'pct': (weighted_i / total_weighted * 100) if total_weighted > 0 else 0.0},
                {'feature': 'E', 'contribution': weighted_e, 'pct': (weighted_e / total_weighted * 100) if total_weighted > 0 else 0.0},
                {'feature': 'R', 'contribution': weighted_r, 'pct': (weighted_r / total_weighted * 100) if total_weighted > 0 else 0.0}
            ], key=lambda x: x['contribution'], reverse=True)[:3]
        }
        
        # Build metadata record
        metadata = {
            'ticker': ticker,
            'name': ranked_row.get('name', ''),
            'exchange': ranked_row.get('exchange', ''),
            'confidence': float(ranked_row.get('confidence_final', 0.0)),  # Prominent confidence field
            'features': {
                'P': p_value,
                'C': c_value,
                'M': m_value,
                'S': s_value,
                'I': i_value,
                'E': e_value,
                'R': r_value
            },
            'rule_score': float(ranked_row.get('ml_score', 0.0)),
            'knn_score': float(candidate_row.get('S_fast', 0.0)) if candidate_row is not None else 0.0,
            'score_linear': float(ranked_row.get('score_linear', 0.0)) if pd.notna(ranked_row.get('score_linear', np.nan)) else None,
            'score_model': float(ranked_row.get('score_model', 0.0)) if pd.notna(ranked_row.get('score_model', np.nan)) else None,
            'shap': {
                'base_value': float(ranked_row.get('shap_base_value', 0.0)) if pd.notna(ranked_row.get('shap_base_value', np.nan)) else None,
                'P': float(ranked_row.get('shap_P', 0.0)) if pd.notna(ranked_row.get('shap_P', np.nan)) else None,
                'C': float(ranked_row.get('shap_C', 0.0)) if pd.notna(ranked_row.get('shap_C', np.nan)) else None,
                'M': float(ranked_row.get('shap_M', 0.0)) if pd.notna(ranked_row.get('shap_M', np.nan)) else None,
                'S': float(ranked_row.get('shap_S', 0.0)) if pd.notna(ranked_row.get('shap_S', np.nan)) else None,
            },
            'explanation': explanation,  # SHAP-like breakdown (rule-based)
            'contributions': {
                'pct_P': float(ranked_row.get('pct_P', 0.0)),
                'pct_C': float(ranked_row.get('pct_C', 0.0)),
                'pct_M': float(ranked_row.get('pct_M', 0.0)),
                'pct_S': float(ranked_row.get('pct_S', 0.0)),
                'pct_I': float(ranked_row.get('pct_I', 0.0)),
                'pct_E': float(ranked_row.get('pct_E', 0.0)),
                'pct_R': float(ranked_row.get('pct_R', 0.0))
            },
            'evidence_snippets': evidence_snippets,  # 2-3 text snippets showing why it's a comp
            'evidence_by_feature': evidence_by_feature,  # Evidence mapped to P, C, M, S features
            'business_model': business_model_info,  # Business model classification and gate
            'economic_signature': extracted.get('economic_signature', {}),  # Economic signature for archetype matching
            'archetype_match': ranked_row.get('archetype_info', {}),  # NEW: Archetype matching info (target_archetype, candidate_archetype, similarity)
            'concept_matches': json.loads(ranked_row.get('concept_matches', '[]')) if isinstance(ranked_row.get('concept_matches'), str) else ranked_row.get('concept_matches', []),
            'segment_mix_target': target.get('segment_mix', {}),
            'segment_mix_candidate': extracted.get('segment_mix', {}),
            'segment_similarity': float(ranked_row.get('sim_cosine', ranked_row.get('S', 0.0))),  # Use sim_cosine from S feature, fallback to S
            'dominant_segment_match': _get_dominant_segment_match(target.get('segment_mix', {}), extracted.get('segment_mix', {})),
            'evidence': extracted.get('evidence', []),  # Keep full evidence for backward compatibility
            'confidence_final': float(ranked_row.get('confidence_final', 0.0)),  # Keep for backward compatibility
            'model_meta': extracted.get('model_meta', {}),
            'paths': candidate_row.get('paths', '') if candidate_row is not None else '',
            'passed_gates': bool(ranked_row.get('passed_gates', False)),
            'gate_business_model': bool(ranked_row.get('gate_business_model', False)),
            'timestamp': datetime.utcnow().isoformat()
        }
        metadata_records.append(metadata)
    
    # Write JSONL
    with open(metadata_path, 'w') as f:
        for record in metadata_records:
            f.write(json.dumps(record) + '\n')
    print(f"✓ Wrote metadata to {metadata_path}")
    
    # Run Summary JSON (part of step 10)
    run_summary = {
        'target_id': target_id,
        'target_name': target.get('name', ''),
        'mode': mode,
        'timestamp': datetime.utcnow().isoformat(),
        'provenance': {
            'config_version': prompt_version,
            'weights': config.get('weights', {}),
            'recall_config': config.get('recall', {})
        },
        'thresholds': {
            'shortlist_cap': config.get('shortlist_cap', 80),
            'tenk_trigger_topN': config.get('tenk_trigger_topN', 30),
            'min_product_hits': config.get('weights', {}).get(mode, {}).get('gates', {}).get('min_product_hits', 2),
            'min_shared_segments': config.get('weights', {}).get(mode, {}).get('gates', {}).get('min_shared_segments', 1)
        },
        'path_contributions': {
            'A': len([r for r in metadata_records if 'A' in r.get('paths', '')]),
            'B': len([r for r in metadata_records if 'B' in r.get('paths', '')]),
            'C': len([r for r in metadata_records if 'C' in r.get('paths', '')]),
            'D': len([r for r in metadata_records if 'D' in r.get('paths', '')])
        },
        'metrics': {
            'total_candidates': len(candidates_df),
            'shortlisted': len(shortlist_df),
            'final_ranked': len(ranked_df),
            'passed_gates': len([r for r in metadata_records if r.get('passed_gates', False)]),
            'avg_confidence': np.mean([r.get('confidence_final', 0.0) for r in metadata_records]) if metadata_records else 0.0
        },
        'cache_stats': {
            'evidence_cache_hits': 0,  # TODO: track cache hits
            'embedding_cache_hits': 0  # TODO: track cache hits
        },
        'errors': []  # TODO: collect errors during run
    }
    
    summary_path = os.path.join(OUTPUTS_DIR, f'{target_id}_run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"✓ Wrote run summary to {summary_path}")
    
    # 11. Final Summary
    print("\nPipeline complete!")
    print("="*80)
    print(f"KNN leaderboard: {knn_path}")
    print(f"Ranked leaderboard: {ranked_path}")
    print(f"Final top 10 comparables: {final_comps_path}")
    print(f"Metadata JSONL: {metadata_path}")
    print(f"Run summary: {summary_path}")
    print(f"Total candidates: {len(candidates_df)}")
    print(f"Shortlisted: {len(shortlist_df)}")
    print(f"Final ranked: {len(ranked_df)}")
    print(f"Passed gates: {run_summary['metrics']['passed_gates']}")
    print("="*80)


def _get_dominant_segment_match(target_mix, candidate_mix):
    """Get dominant segment match between target and candidate."""
    if not target_mix or not candidate_mix:
        return False
    
    # Find dominant segment in target
    target_dominant = max(target_mix.items(), key=lambda x: x[1])[0] if target_mix else None
    candidate_dominant = max(candidate_mix.items(), key=lambda x: x[1])[0] if candidate_mix else None
    
    return target_dominant == candidate_dominant


if __name__ == "__main__":
    main()
