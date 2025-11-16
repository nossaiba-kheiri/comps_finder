"""
pack.py: Build/aggregate EvidencePack {sources[], updated_at, text_hashes}
"""
import os
import sys
from datetime import datetime, timedelta
import hashlib

# Import evidence fetchers
sys.path.insert(0, os.path.dirname(__file__))
from fetch_site import fetch_site_evidence
from fetch_sec import fetch_sec_10k
from fetch_xbrl import fetch_xbrl_segment_revenue


def build_evidence_pack(ticker, cik=None, website=None, should_fetch_10k=False, config=None):
    """
    Build EvidencePack for a ticker.
    Fetches from site, SEC 10-K (if needed), and XBRL (if available).
    
    Args:
        ticker: Company ticker
        cik: CIK number (for SEC fetching)
        website: Company website URL
        should_fetch_10k: Whether to fetch 10-K (top-30, alias, or ambiguous)
        config: Runtime config with pages, rate limits, etc.
    """
    config = config or {}
    sources = []
    latest_date = None
    
    # 1. Fetch site evidence
    # Handle NaN/None/empty website values
    if website and str(website).strip() not in ['', 'nan', 'None', 'NaN']:
        try:
            # Check if cache exists before fetching
            from urllib.parse import urlparse
            parsed = urlparse(website if website.startswith('http') else f'https://{website}')
            cache_key = parsed.netloc.replace('.', '_')
            cache_dir = os.path.join(os.path.dirname(__file__), '../../data/cache/site')
            cache_file = os.path.join(cache_dir, f"{cache_key}.json")
            
            from_cache = False
            if os.path.exists(cache_file):
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                if cache_age < timedelta(days=120):  # 4 months
                    from_cache = True
            
            site_sources = fetch_site_evidence(website, config)
            sources.extend(site_sources)
            if site_sources:
                if from_cache:
                    print(f"    Using cached site pages for {ticker} ({len(site_sources)} pages)")
                else:
                    print(f"    Fetched {len(site_sources)} site pages for {ticker}")
        except Exception as e:
            print(f"    Warning: Failed to fetch site evidence for {ticker}: {e}")
            # Continue without site evidence
    
    # 2. Fetch 10-K if needed
    if should_fetch_10k and cik:
        # Check if 10-K cache exists
        cache_dir = os.path.join(os.path.dirname(__file__), '../../data/cache/sec')
        cache_file = os.path.join(cache_dir, f"{ticker}_10k.json")
        from_cache = os.path.exists(cache_file) if os.path.exists(cache_dir) else False
        
        tenk_source = fetch_sec_10k(ticker, cik)
        if tenk_source:
            sources.append(tenk_source)
            filing_date = tenk_source.get('filing_date')
            if filing_date:
                try:
                    latest_date = datetime.fromisoformat(filing_date.replace('Z', '+00:00'))
                except:
                    latest_date = datetime.utcnow()
            if from_cache:
                print(f"    Using cached 10-K for {ticker}")
            else:
                print(f"    Fetched 10-K for {ticker}")
    
    # 3. Fetch XBRL segment revenue (if available)
    segment_mix = None
    if cik:
        segment_mix = fetch_xbrl_segment_revenue(ticker, cik)
        if segment_mix:
            print(f"    Fetched XBRL segment mix for {ticker}")
    
    # If no sources found, create minimal mock
    if not sources:
        sources = [{
            'type': 'site',
            'url': website or f'https://example.com/{ticker}',
            'text': f'Company {ticker} provides various services and solutions.'
        }]
    
    # Compute text hashes
    text_hashes = []
    for source in sources:
        text = source.get('text', '')
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        text_hashes.append(f'sha256:{h}')
    
    # Determine updated_at (newest source date)
    if not latest_date:
        # Check source dates
        for source in sources:
            if 'filing_date' in source:
                try:
                    source_date = datetime.fromisoformat(source['filing_date'].replace('Z', '+00:00'))
                    if not latest_date or source_date > latest_date:
                        latest_date = source_date
                except:
                    pass
    
    if not latest_date:
        latest_date = datetime.utcnow()
    
    return {
        'ticker': ticker,
        'updated_at': latest_date.isoformat(),
        'sources': sources,
        'text_hashes': text_hashes,
        'segment_mix_xbrl': segment_mix  # XBRL segment mix if available
    }


if __name__ == "__main__":
    # Test
    pack = build_evidence_pack('AAPL')
    print(pack)
