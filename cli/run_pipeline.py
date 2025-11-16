"""
run_pipeline.py: Main orchestrator for the company comparator pipeline.

Supports two modes:
1. Use existing target.json: --target data/target.json
2. Create target.json from basic info: --name, --url, --description, --primary-industry-classification
"""
import os
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
from features.segment_mix import score_segment_mix
from features.semantic import score_semantic_similarity
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
    # P: Product overlap (materiality-weighted, using NLP embeddings)
    # Use products field if available, fallback to business_activity
    target_products = target.get('products', [])
    if not target_products:  # If products is missing or empty, use business_activity as fallback
        target_products = target.get('business_activity', [])
    candidate_products = extracted_data.get('business_activity', [])
    initiatives = extracted_data.get('initiatives', [])
    P, product_hits, concept_matches = score_product_overlap(
        target_products, candidate_products, initiatives=initiatives, run_with_openai=run_with_openai
    )
    
    # C: Customer overlap
    # Use customer_segment if available, fallback to customers field
    target_customers = target.get('customer_segment', target.get('customers', []))
    candidate_customers = extracted_data.get('customer_segment', [])
    C, customer_hits = score_customer_overlap(target_customers, candidate_customers)
    
    # M: Segment mix
    target_mix = target.get('segment_mix', {})
    candidate_mix = extracted_data.get('segment_mix', {})
    # Use XBRL segment mix if available
    if not candidate_mix and evidence_pack.get('segment_mix_xbrl'):
        candidate_mix = evidence_pack.get('segment_mix_xbrl')
    M = score_segment_mix(target_mix, candidate_mix) if target_mix and candidate_mix else 0.5
    
    # S: Semantic similarity
    # Use raw_profile_text if available (contains website + LinkedIn), otherwise use text_profile or construct
    target_text = target.get('raw_profile_text') or target.get('text_profile', '')
    if not target_text:
        # Construct from structured fields as fallback
        target_products = target.get('business_activity', []) or target.get('products', [])
        target_customers = target.get('customer_segment', []) or target.get('customers', [])
        target_text = ' '.join(target_products + target_customers)
    
    candidate_text = ' '.join(candidate_products + candidate_customers)
    try:
        target_emb = get_cached_embedding(target_text, run_with_openai=run_with_openai)
        candidate_emb = get_cached_embedding(candidate_text, run_with_openai=run_with_openai)
        S = score_semantic_similarity(target_emb, candidate_emb)
    except Exception as e:
        # Fallback to S_fast from candidate generation
        S = candidate_row.get('S_fast', 0.0) if 'S_fast' in candidate_row else 0.0
    
    # I: Industry proximity
    # CRITICAL: Compare customer industries (verticals served), not own industry
    # Both target and candidate should be in the same own industry (filtered earlier)
    # We compare which customer industries each serves
    
    target_own_industry = target.get('primary_industry_classification', '').lower()
    candidate_own_industry = str(candidate_row.get('industry', '')).lower()
    
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
    
    # E: Evidence quality
    sources = evidence_pack.get('sources', [])
    E = score_evidence_quality(sources)
    
    # R: Recency
    updated_at = evidence_pack.get('updated_at', '')
    R = score_recency(updated_at)
    
    # Confidence (blended: LLM confidence + evidence coverage + E score)
    llm_confidence = extracted_data.get('confidence_0_1', 0.0)
    evidence_coverage = min(len(sources) / 5.0, 1.0)  # Normalize to [0, 1]
    confidence_final = 0.4 * llm_confidence + 0.3 * evidence_coverage + 0.3 * E
    
    return {
        'P': P,
        'C': C,
        'M': M,
        'S': S,
        'I': I,
        'E': E,
        'R': R,
        'product_hits': product_hits,
        'customer_hits': customer_hits,
        'LLM_confidence': llm_confidence,
        'confidence_final': confidence_final,
        'concept_matches': concept_matches,
        'segment_mix': candidate_mix,
        'initiatives': initiatives
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
    
    # Pipeline options
    parser.add_argument('--openai', action='store_true', help='Use real OpenAI embeddings and LLM')
    parser.add_argument('--limit-candidates', type=int, default=None, help='Limit number of candidates for testing')
    parser.add_argument('--force', action='store_true', help='Force recreation of target.json even if cached version exists')
    
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
                api_key=os.getenv('OPENAI_API_KEY') if args.openai else None
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
    target_id = target.get('name', 'target').replace(' ', '_').lower()
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
    
    # 4. EvidencePack gathering (real fetching with 10-K logic)
    print("\n[4/10] Gathering evidence...")
    evidence_packs = {}
    tenk_trigger_topN = config.get('tenk_trigger_topN', 30)
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
        
        # Determine if we should fetch 10-K:
        # - Top-30 by rank_key, OR
        # - Segment alias hit (path 'D'), OR
        # - Site evidence ambiguous (we'll check after site fetch)
        should_fetch_10k = (
            idx < tenk_trigger_topN or
            'D' in paths or
            rank_key < 0.3  # Low rank_key might indicate ambiguous evidence
        )
        
        evidence_packs[ticker] = build_evidence_pack(
            ticker=ticker,
            cik=cik,
            website=website,
            should_fetch_10k=should_fetch_10k,
            config=config
        )
    print(f"✓ Gathered evidence for {len(evidence_packs)} candidates")
    
    # 5. LLM extraction
    print("\n[5/10] Extracting structured data with LLM...")
    extracted_data = {}
    prompt_version = config.get('prompt_version', 'svc_cust_v3')
    run_with_llm = args.openai  # Use LLM if OpenAI flag is set
    for ticker, pack in evidence_packs.items():
        extracted_data[ticker] = extract_llm_structured(
            pack, 
            prompt_version=prompt_version,
            run_with_llm=run_with_llm
        )
    print(f"✓ Extracted data for {len(extracted_data)} candidates")
    
    # 6. Feature computation
    print("\n[6/10] Computing features...")
    feature_rows = []
    for _, candidate_row in shortlist_df.iterrows():
        ticker = candidate_row['ticker']
        pack = evidence_packs.get(ticker, {})
        extracted = extracted_data.get(ticker, {})
        
        features = compute_features(target, candidate_row, extracted, pack, run_with_openai=args.openai)
        
        # Convert concept_matches to JSON string for CSV
        concept_matches_json = json.dumps(features.get('concept_matches', []))
        initiatives_json = json.dumps(features.get('initiatives', []))
        
        feature_row = {
            'ticker': ticker,
            'name': candidate_row.get('name', ''),
            'exchange': candidate_row.get('exchange', ''),
            'P': features['P'],
            'C': features['C'],
            'M': features['M'],
            'S': features['S'],
            'I': features['I'],
            'E': features['E'],
            'R': features['R'],
            'product_hits': features['product_hits'],
            'customer_hits': features['customer_hits'],
            'LLM_confidence': features['LLM_confidence'],
            'confidence_final': features['confidence_final'],
            'concept_matches': concept_matches_json,
            'initiatives': initiatives_json,
            'business_activity': ', '.join(extracted.get('business_activity', [])),
            'customer_segment': ', '.join(extracted.get('customer_segment', [])),
            'segment_mix': json.dumps(extracted.get('segment_mix', {})),
            'evidence_urls': '; '.join([s.get('url', '') for s in pack.get('sources', [])]),
            'evidence_quotes': extracted.get('evidence', [{}])[0].get('quote', '') if extracted.get('evidence') else '',
            'prompt_version': prompt_version
        }
        feature_rows.append(feature_row)
    
    features_df = pd.DataFrame(feature_rows)
    print(f"✓ Computed features for {len(features_df)} candidates")
    
    # 6.5. Fit explainer model + compute SHAP values
    print("\n[7/10] Fitting explainer model + computing SHAP values...")
    
    # ALWAYS compute score_linear (needed for ranking even if SHAP fails)
    from ranker.scorer_rule import load_weights_config
    weights_config = load_weights_config(mode)
    feature_weights = weights_config.get('weights', {})
    
    # Extract weights for P, C, M, S (main features for SHAP)
    w_p = feature_weights.get('P', 0.28)
    w_c = feature_weights.get('C', 0.28)
    w_m = feature_weights.get('M', 0.18)
    w_s = feature_weights.get('S', 0.16)
    
    # Compute linear score (pseudo-label) from P, C, M, S
    features_df['score_linear'] = (
        w_p * features_df['P'] +
        w_c * features_df['C'] +
        w_m * features_df['M'] +
        w_s * features_df['S']
    )
    
    # Try to compute SHAP values (optional - for explainability)
    try:
        # Fix OpenMP conflict on macOS (XGBoost/FAISS/NumPy all use OpenMP)
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        import xgboost as xgb
        import shap
        
        # Prepare feature matrix and target
        feature_cols = ['P', 'C', 'M', 'S']
        
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
        y = features_df['score_linear'].values.astype(np.float64)
        
        # Validate shapes
        if len(X) == 0:
            raise ValueError("Empty feature matrix")
        if len(y) == 0:
            raise ValueError("Empty target vector")
        if X.shape[1] != 4:
            raise ValueError(f"Expected 4 features, got {X.shape[1]}")
        
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
        shap_values = explainer.shap_values(X)  # shape: (n_samples, 4)
        
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
        
        print(f"✓ Computed SHAP values for {len(features_df)} candidates")
        print(f"  Model score correlation with linear: {np.corrcoef(features_df['score_linear'], features_df['score_model'])[0,1]:.3f}")
        
    except ImportError as e:
        print(f"  Warning: SHAP/XGBoost not available ({e}). Skipping SHAP computation.")
        # Add placeholder columns with NaN (score_linear already computed above)
        for col in ['P', 'C', 'M', 'S']:
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
            # Set environment variable and retry
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            try:
                # Retry SHAP computation
                import xgboost as xgb
                import shap
                
                # Prepare feature matrix and target (with validation)
                feature_cols = ['P', 'C', 'M', 'S']
                
                # Validate and clean data
                for col in feature_cols:
                    if features_df[col].isna().any():
                        features_df[col] = features_df[col].fillna(0.0)
                    if np.isinf(features_df[col]).any():
                        features_df.loc[np.isinf(features_df[col]), col] = 0.0
                
                X = features_df[feature_cols].values.astype(np.float64)
                y = features_df['score_linear'].values.astype(np.float64)
                
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
                shap_values = explainer.shap_values(X)  # shape: (n_samples, 4)
                
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
                print(f"  Model score correlation with linear: {np.corrcoef(features_df['score_linear'], features_df['score_model'])[0,1]:.3f}")
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
    export_leaderboard(knn_df, knn_path, 'knn')
    
    # 9. Rule-based ranking and gates
    print("\n[9/10] Computing rule-based scores and applying gates...")
    scored_rows = []
    for _, feature_row in features_df.iterrows():
        features_dict = {
            'P': feature_row['P'],
            'C': feature_row['C'],
            'M': feature_row['M'],
            'S': feature_row['S'],
            'I': feature_row['I'],
            'E': feature_row['E'],
            'R': feature_row['R'],
            'product_hits': int(feature_row.get('product_hits', 0)),
            'customer_hits': int(feature_row.get('customer_hits', 0))
        }
        
        score, pct_dict, passed_gates = rule_score(features_dict, mode=mode)
        
        # For testing, include all candidates even if gates fail (but mark it)
        scored_row = {
            **feature_row,
            'ml_score': score,  # Using rule score as ml_score for now
            **pct_dict,
            'passed_gates': passed_gates
        }
        # Include SHAP columns if available
        if 'shap_P' in feature_row:
            scored_row['score_linear'] = feature_row.get('score_linear', np.nan)
            scored_row['score_model'] = feature_row.get('score_model', np.nan)
            scored_row['shap_P'] = feature_row.get('shap_P', np.nan)
            scored_row['shap_C'] = feature_row.get('shap_C', np.nan)
            scored_row['shap_M'] = feature_row.get('shap_M', np.nan)
            scored_row['shap_S'] = feature_row.get('shap_S', np.nan)
            scored_row['shap_base_value'] = feature_row.get('shap_base_value', np.nan)
        scored_rows.append(scored_row)
    
    ranked_df = pd.DataFrame(scored_rows)
    if len(ranked_df) > 0:
        ranked_df = ranked_df.sort_values('ml_score', ascending=False)
        ranked_df['rank_ml'] = range(1, len(ranked_df) + 1)
        
        # Filter to only passed gates for final output (or include all for testing)
        # ranked_df = ranked_df[ranked_df['passed_gates'] == True].copy()
        # ranked_df['rank_ml'] = range(1, len(ranked_df) + 1)
    else:
        ranked_df['rank_ml'] = []
        ranked_df['ml_score'] = []
    
    ranked_path = os.path.join(OUTPUTS_DIR, f'{target_id}_ranked.csv')
    export_leaderboard(ranked_df, ranked_path, 'ranked')
    print(f"✓ Ranked {len(ranked_df)} candidates")
    
    # 9.5. Export final top 10 comparables CSV (clean format)
    print("\n[9.5/10] Exporting final top 10 comparables CSV...")
    top_10_df = ranked_df.head(10).copy()
    
    # Load universe data to get website URLs
    universe_path = os.path.join(DATA_DIR, 'universe_us.csv')
    universe_df = None
    if os.path.exists(universe_path):
        try:
            universe_df = pd.read_csv(universe_path)
        except Exception as e:
            print(f"  Warning: Could not load universe.csv: {e}")
    
    # Build final comps dataframe with required fields
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
            if isinstance(activity_list, list):
                business_activity = ', '.join(activity_list)
            else:
                business_activity = str(activity_list) if activity_list else ''
        
        # Get customer_segment from row (already a joined string, or from extracted_data)
        customer_segment = row.get('customer_segment', '')
        if not customer_segment or customer_segment == '':
            # Try to get from extracted_data
            extracted = extracted_data.get(ticker, {})
            segment_list = extracted.get('customer_segment', [])
            if isinstance(segment_list, list):
                customer_segment = ', '.join(segment_list)
            else:
                customer_segment = str(segment_list) if segment_list else ''
        
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
    final_comps_path = os.path.join(OUTPUTS_DIR, f'{target_id}_final_comps.csv')
    final_comps_df.to_csv(final_comps_path, index=False)
    print(f"✓ Exported final top {len(final_comps_df)} comparables to: {final_comps_path}")
    
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
        # Load weights from weights.yaml (same as rule_score uses)
        from ranker.scorer_rule import load_weights_config
        weights_config = load_weights_config(mode)
        feature_weights = weights_config.get('weights', {})
        
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
            'concept_matches': json.loads(ranked_row.get('concept_matches', '[]')) if isinstance(ranked_row.get('concept_matches'), str) else ranked_row.get('concept_matches', []),
            'segment_mix_target': target.get('segment_mix', {}),
            'segment_mix_candidate': extracted.get('segment_mix', {}),
            'segment_similarity': float(ranked_row.get('M', 0.0)),
            'dominant_segment_match': _get_dominant_segment_match(target.get('segment_mix', {}), extracted.get('segment_mix', {})),
            'evidence': extracted.get('evidence', []),  # Keep full evidence for backward compatibility
            'confidence_final': float(ranked_row.get('confidence_final', 0.0)),  # Keep for backward compatibility
            'model_meta': extracted.get('model_meta', {}),
            'paths': candidate_row.get('paths', '') if candidate_row is not None else '',
            'passed_gates': bool(ranked_row.get('passed_gates', False)),
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
