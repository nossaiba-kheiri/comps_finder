"""
run_pipeline2.py: LLM-based ranking pipeline

This pipeline uses an LLM to directly select the top 10 comparables after shortlisting.
Instead of rule-based scoring, it presents all candidate data to the LLM and asks it to rank/select.
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
ROOT = str(PROJECT_ROOT)

# Add src to path for imports (must be before imports)
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# Directories
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = DATA_DIR / 'outputs'
CONFIG_DIR = PROJECT_ROOT / 'config'

# Import modules (after path setup)
import yaml
from universe.generate_candidates import generate_candidates
from prelim.prelim_filter import prelim_filter
from evidence.pack import build_evidence_pack
from nlp.llm_extract import extract_llm_structured

# Helper functions (from run_pipeline.py)
def load_config():
    """Load runtime configuration."""
    runtime_path = CONFIG_DIR / 'runtime.yaml'
    if runtime_path.exists():
        with open(runtime_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_target(target_path):
    """Load target JSON."""
    with open(target_path, 'r') as f:
        return json.load(f)

# Ensure output directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False


def format_candidate_for_llm(ticker, name, extracted, evidence_pack, candidate_row):
    """
    Format candidate data for LLM prompt.
    Extracts key information in a structured way.
    """
    # Get business activity (filter out generic terms)
    business_activity = extracted.get('business_activity', [])
    business_activity = [ba for ba in business_activity if ba.lower() not in ['services', 'solutions', 'products']]
    
    # Get customer segments
    customer_segment = extracted.get('customer_segment', [])
    primary_customer_types = extracted.get('primary_customer_types', [])
    
    # Get business model info
    business_model_type = extracted.get('business_model_type', 'unknown')
    services_share = extracted.get('services_share_estimate', 0.5)
    revenue_model = extracted.get('revenue_model', [])
    
    # Get segment mix
    segment_mix = extracted.get('segment_mix', {})
    
    # Get evidence quotes (most relevant)
    evidence_quotes = []
    if evidence_pack:
        sources = evidence_pack.get('sources', [])
        for source in sources[:3]:  # Top 3 sources
            if source.get('text'):
                # Take first 200 chars of text
                text = source.get('text', '')[:200]
                if text:
                    evidence_quotes.append(text)
    
    # Format as structured text
    candidate_text = f"""
Ticker: {ticker}
Name: {name}
Exchange: {candidate_row.get('exchange', 'N/A')}

Business Model:
- Type: {business_model_type}
- Services Share: {services_share:.0%}
- Revenue Model: {', '.join(revenue_model) if revenue_model else 'N/A'}

Business Activities:
{chr(10).join(f'  - {ba}' for ba in business_activity[:5]) if business_activity else '  - N/A'}

Customer Segments:
{chr(10).join(f'  - {cs}' for cs in customer_segment[:5]) if customer_segment else '  - N/A'}

Segment Mix (Revenue Distribution):
{json.dumps(segment_mix, indent=2) if segment_mix else '  - N/A'}

Key Evidence:
{chr(10).join(f'  - {quote}...' for quote in evidence_quotes[:2]) if evidence_quotes else '  - N/A'}
"""
    return candidate_text


def ask_llm_to_rank_comps(target, candidates_data, api_key=None):
    """
    Ask LLM to select and rank the top 10 comparables.
    
    Args:
        target: Target company dict
        candidates_data: List of formatted candidate strings
        api_key: OpenAI API key
    
    Returns:
        List of tickers in ranked order (top 10)
    """
    if not openai_available:
        print("  ERROR: OpenAI not available. Cannot use LLM ranking.")
        return []
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ERROR: OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return []
    
    client = OpenAI(api_key=api_key)
    
    # Format target information
    target_name = target.get('name', 'Target Company')
    target_business = ', '.join(target.get('business_activity', [])[:5])
    target_customers = ', '.join(target.get('customer_segment', [])[:5])
    target_bm_type = target.get('business_model_type', 'unknown')
    target_services_share = target.get('services_share_estimate', 0.5)
    target_segment_mix = target.get('product_mix', {}) or target.get('segment_mix', {})
    
    # Get target's primary industry and similar industries for context
    target_primary_industry = target.get('primary_industry_classification', 'N/A')
    target_similar_industries = target.get('similar_industries', [])
    similar_industries_str = ', '.join(target_similar_industries) if target_similar_industries else 'N/A'
    
    # Build prompt
    prompt = f"""You are a financial analyst selecting comparable companies (comps) for valuation and benchmarking.

TARGET COMPANY:
Name: {target_name}
Primary Industry: {target_primary_industry}
Similar Industries (in priority order): {similar_industries_str}
Business Activities: {target_business}
Customer Segments: {target_customers}
Business Model Type: {target_bm_type}
Services Share: {target_services_share:.0%}
Segment Mix: {json.dumps(target_segment_mix, indent=2) if target_segment_mix else 'N/A'}

CANDIDATE COMPANIES:
{chr(10).join(f'{i+1}. {candidate}' for i, candidate in enumerate(candidates_data))}

TASK:
Select the top 10 companies that are the BEST comparables for {target_name}.

Selection Criteria (in order of importance):
1. INDUSTRY MATCH (HIGHEST PRIORITY): Companies in the same PRIMARY INDUSTRY or first similar industry are the best comps
   - Prioritize companies whose industry matches "{target_primary_industry}" or "{target_similar_industries[0] if target_similar_industries else 'N/A'}"
   - Companies serving client industries are NOT companies in those client industries - they are companies in their own industry
   - A company's industry is what the company IS, not what industries its clients are in
2. Business Model Similarity: Similar revenue model (services vs software vs hybrid), similar services share
3. Economic Similarity: Similar delivery model, deal structure, buyer persona, transformation intent
4. Segment Mix Similarity: Similar customer segment distribution (portfolio-aware, not just single-segment match)
5. Product/Service Similarity: Similar offerings and capabilities

CRITICAL RULES:
- INDUSTRY MATCH IS THE PRIMARY CRITERIA: Companies in the same industry (especially "{target_similar_industries[0] if target_similar_industries else target_primary_industry}") rank highest
- DO NOT confuse CUSTOMER INDUSTRIES with COMPANY INDUSTRY:
  * If target serves client industries but is in a specific company industry → prioritize other companies in that same company industry
  * If target has client industry terms in products but is in a different company industry → prioritize companies in the target's company industry, not companies in the client industry
- Reject companies that are clearly wrong industry
- For service-based targets: prioritize companies in the same industry over technology companies that happen to serve similar clients

Return ONLY a JSON array of exactly 10 tickers in ranked order (best comp first):
["TICKER1", "TICKER2", "TICKER3", ..., "TICKER10"]

If there are fewer than 10 suitable candidates, include only the suitable ones (but try to find 10).
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert at selecting comparable companies. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent ranking
            max_tokens=500
        )
        
        # Parse response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON array
        # Remove markdown code blocks if present
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])  # Remove first and last lines (``` markers)
        
        # Parse JSON
        ranked_tickers = json.loads(content)
        
        if not isinstance(ranked_tickers, list):
            print(f"  WARNING: LLM returned non-list: {type(ranked_tickers)}")
            return []
        
        return ranked_tickers[:10]  # Ensure max 10
        
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse LLM response as JSON: {e}")
        print(f"  Response: {content[:200]}")
        return []
    except Exception as e:
        print(f"  ERROR: LLM ranking failed: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Company Comparator Pipeline 2 (LLM-based ranking)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing target.json
  python cli/run_pipeline2.py --target data/target_huron_consulting_group_inc.json --openai
  
  # Create target.json from basic info and run pipeline
  python cli/run_pipeline2.py \\
    --name "Company Name" \\
    --url "https://company.com" \\
    --description "Business description..." \\
    --primary-industry-classification "Industry Name" \\
    --openai
        """
    )
    
    # Target input options
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
    parser.add_argument('--ticker', type=str, help='Company ticker symbol (optional)')
    
    # Pipeline options
    parser.add_argument('--openai', action='store_true', help='Use real OpenAI embeddings and LLM')
    parser.add_argument('--limit-candidates', type=int, default=None, help='Limit number of candidates for testing')
    parser.add_argument('--force', action='store_true', help='Force recreation of target.json even if cached version exists')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Company Comparator Pipeline 2 (LLM-based ranking)")
    print("="*80)
    
    # 0. Create target.json if basic info provided
    target_path = args.target
    if args.name:
        if not args.url or not args.description or not args.primary_industry_classification:
            parser.error("When using --name, --url, --description, and --primary-industry-classification are required")
        
        safe_name = args.name.replace(' ', '_').replace('/', '_').lower()
        target_path = os.path.join(DATA_DIR, f'target_{safe_name}.json')
        
        if os.path.exists(target_path) and not args.force:
            print("\n[0/6] Loading existing target.json from cache...")
            print(f"✓ Found cached target.json: {target_path}")
        else:
            print("\n[0/6] Creating target.json from input data...")
            import sys
            sys.path.insert(0, str(DATA_DIR))
            from create_target_from_info import create_target_from_info
            
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
            
            with open(target_path, 'w') as f:
                json.dump(target_data, f, indent=2)
            print(f"✓ Created and saved target.json: {target_path}")
    
    # 1. Load configs and target
    print("\n[1/6] Loading configs and target...")
    config = load_config()
    target = load_target(target_path)
    target_id = target.get('name', 'target').replace(' ', '_').lower()
    print(f"✓ Loaded target: {target.get('name')}")
    
    # 2. Generate candidates
    print("\n[2/6] Generating candidates...")
    candidates_df = generate_candidates(target, config, run_with_openai=args.openai)
    if args.limit_candidates:
        candidates_df = candidates_df.head(args.limit_candidates)
    print(f"✓ Generated {len(candidates_df)} candidates")
    
    # 3. Shortlist
    print("\n[3/6] Creating shortlist...")
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
        industry = row.get('industry', 'N/A')[:30]
        print(f"   {idx:2d}. {ticker:6s} - {name:50s} | rank_key: {rank_key:.3f} | paths: {paths} | {industry}")
    
    # Save shortlist to file for inspection
    shortlist_path = os.path.join(OUTPUTS_DIR, f'{target_id}_shortlist.csv')
    shortlist_df.to_csv(shortlist_path, index=False)
    print(f"\n💾 Saved shortlist to: {shortlist_path}")
    
    # 4. Evidence gathering
    print("\n[4/6] Gathering evidence...")
    evidence_packs = {}
    tenk_trigger_topN = config.get('tenk_trigger_topN', 30)
    total_candidates = len(shortlist_df)
    processed = 0
    
    for idx, row in shortlist_df.iterrows():
        ticker = row['ticker']
        cik = row.get('cik', '')
        website = row.get('website', '')
        if pd.isna(website) or (isinstance(website, float) and np.isnan(website)):
            website = ''
        elif not isinstance(website, str):
            website = str(website).strip() if website else ''
        else:
            website = website.strip() if website else ''
        
        rank_key = row.get('rank_key', 0.0)
        should_fetch_10k = (idx < tenk_trigger_topN)
        
        evidence_config = (config or {}).copy()
        evidence_config['should_fetch_xbrl'] = True
        evidence_config['should_fetch_xbrl_if_not_cached'] = (idx < 30)
        
        evidence_packs[ticker] = build_evidence_pack(
            ticker=ticker,
            cik=cik,
            website=website,
            should_fetch_10k=should_fetch_10k,
            config=evidence_config
        )
        
        processed += 1
        if processed % 10 == 0 or processed == total_candidates:
            print(f"    Progress: {processed}/{total_candidates} companies processed")
    print(f"✓ Gathered evidence for {len(evidence_packs)} candidates")
    
    # 5. LLM extraction
    print("\n[5/6] Extracting structured data with LLM...")
    extracted_data = {}
    prompt_version = config.get('prompt_version', 'svc_cust_v3')
    run_with_llm = args.openai
    
    # Track cache usage
    cache_hits = 0
    cache_misses = 0
    total_candidates = len([p for p in evidence_packs.values() if p])
    processed = 0
    
    print(f"    Processing {total_candidates} companies (cache enabled)...")
    
    import time
    start_time = time.time()
    last_progress_time = start_time
    
    for ticker, pack in evidence_packs.items():
        if not pack:
            continue
        
        processed += 1
        
        # Show progress more frequently (every 5 companies or every 5 seconds)
        current_time = time.time()
        if processed % 5 == 0 or (current_time - last_progress_time) >= 5.0:
            elapsed = current_time - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            hit_rate = (cache_hits / processed * 100) if processed > 0 else 0
            remaining = total_candidates - processed
            eta = (remaining / rate) if rate > 0 else 0
            print(f"    [{processed}/{total_candidates}] Cache: {cache_hits}✓ {cache_misses}✗ ({hit_rate:.0f}% hit) | {rate:.1f}/sec | ETA: {eta:.0f}s")
            last_progress_time = current_time
        
        # Call extraction (cache check happens inside this function - no double check!)
        company_start = time.time()
        extracted = extract_llm_structured(
            pack,
            api_key=os.getenv('OPENAI_API_KEY') if run_with_llm else None,
            prompt_version=prompt_version,
            run_with_llm=run_with_llm,
            use_cache=True
        )
        company_time = time.time() - company_start
        
        # Track cache hits/misses by timing (cache hits are <0.1s, misses are 2-5s)
        if company_time < 0.5:
            cache_hits += 1
        else:
            cache_misses += 1
            # Show slow companies
            if company_time > 3.0:
                print(f"      ⚠ {ticker} took {company_time:.1f}s (LLM call)")
        
        if extracted:
            extracted_data[ticker] = extracted
    
    elapsed_total = time.time() - start_time
    cache_hit_rate = (cache_hits / total_candidates * 100) if total_candidates > 0 else 0
    avg_time = elapsed_total / total_candidates if total_candidates > 0 else 0
    print(f"✓ Extracted structured data for {len(extracted_data)} candidates")
    print(f"  Cache: {cache_hits} hits, {cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
    print(f"  Time: {elapsed_total:.1f}s total ({avg_time:.2f}s per company)")
    
    if cache_hit_rate < 50 and run_with_llm:
        print(f"  ⚠ Low cache hit rate - many companies may be new or evidence changed")
        print(f"  💡 Tip: Re-running will be much faster as cache builds up")
    
    # 6. LLM-based ranking
    print("\n[6/6] Asking LLM to select top 10 comparables...")
    
    # Format all candidates for LLM
    candidates_formatted = []
    ticker_to_data = {}
    
    for idx, row in shortlist_df.iterrows():
        ticker = row['ticker']
        name = row.get('name', ticker)
        
        extracted = extracted_data.get(ticker, {})
        pack = evidence_packs.get(ticker, {})
        
        if not extracted:
            # Skip candidates without LLM extraction
            continue
        
        candidate_text = format_candidate_for_llm(ticker, name, extracted, pack, row)
        candidates_formatted.append(candidate_text)
        ticker_to_data[ticker] = {
            'name': name,
            'row': row,
            'extracted': extracted,
            'pack': pack
        }
    
    print(f"  Formatted {len(candidates_formatted)} candidates for LLM")
    
    # Ask LLM to rank
    if not args.openai:
        print("  WARNING: --openai flag not set. Cannot use LLM ranking.")
        print("  Falling back to simple ranking by rank_key...")
        ranked_tickers = shortlist_df.head(10)['ticker'].tolist()
    else:
        ranked_tickers = ask_llm_to_rank_comps(
            target,
            candidates_formatted,
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    if not ranked_tickers:
        print("  WARNING: LLM returned no tickers. Falling back to rank_key...")
        ranked_tickers = shortlist_df.head(10)['ticker'].tolist()
    
    print(f"✓ LLM selected {len(ranked_tickers)} comparables")
    
    # 7. Create output
    print("\n[7/7] Creating output files...")
    
    # Build final comps DataFrame
    final_comps = []
    for rank, ticker in enumerate(ranked_tickers, 1):
        if ticker not in ticker_to_data:
            continue
        
        data = ticker_to_data[ticker]
        row = data['row']
        extracted = data['extracted']
        
        final_comps.append({
            'rank': rank,
            'ticker': ticker,
            'name': data['name'],
            'exchange': row.get('exchange', ''),
            'url': row.get('website', ''),
            'business_activity': ', '.join(extracted.get('business_activity', [])[:3]),
            'customer_segment': ', '.join(extracted.get('customer_segment', [])[:3]),
            'business_model_type': extracted.get('business_model_type', 'unknown'),
            'services_share_estimate': extracted.get('services_share_estimate', 0.5),
            'sic_industry': row.get('industry', '')
        })
    
    final_comps_df = pd.DataFrame(final_comps)
    
    # Save outputs
    output_prefix = f'{target_id}_pipeline2'
    
    # Final comps CSV
    final_comps_path = OUTPUTS_DIR / f'{output_prefix}_final_comps.csv'
    final_comps_df.to_csv(final_comps_path, index=False)
    print(f"✓ Saved final comps to {final_comps_path}")
    
    # Summary JSON
    summary = {
        'target_id': target_id,
        'target_name': target.get('name'),
        'pipeline': 'pipeline2_llm_ranking',
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_candidates': len(shortlist_df),
        'candidates_with_extraction': len(extracted_data),
        'final_comps_count': len(final_comps),
        'ranked_tickers': ranked_tickers
    }
    
    summary_path = OUTPUTS_DIR / f'{output_prefix}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")
    
    # Print top 10
    print("\n" + "="*80)
    print("TOP 10 COMPARABLES (LLM-selected):")
    print("="*80)
    for comp in final_comps:
        print(f"{comp['rank']:2d}. {comp['ticker']:6s} - {comp['name']}")
        print(f"     Business Model: {comp['business_model_type']}, Services: {comp['services_share_estimate']:.0%}")
        print(f"     Activities: {comp['business_activity'][:80]}...")
        print()
    
    print("="*80)
    print("✓ Pipeline 2 complete!")
    print(f"✓ Output files saved to {OUTPUTS_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()

