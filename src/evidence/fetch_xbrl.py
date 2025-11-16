"""
fetch_xbrl.py: Download/read XBRL segment revenues for a ticker (if available).
"""
import os
import requests
import time
import json

SEC_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../data/cache/xbrl')


def fetch_xbrl_segment_revenue(ticker, cik):
    """
    Fetch XBRL segment revenue disaggregation for a ticker.
    Returns dict with segment_mix {segment: revenue_share} or None.
    """
    if not cik or not ticker:
        return None
    
    # Clean CIK
    cik_clean = str(cik).lstrip('0')
    if not cik_clean:
        return None
    
    try:
        # Check cache
        cache_file = os.path.join(CACHE_DIR, f"{ticker}_xbrl.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                return cached.get('segment_mix')
        
        # Fetch company facts
        url = SEC_COMPANY_FACTS.format(cik=cik_clean)
        headers = {'User-Agent': 'CompFinder/1.0 you@email.com'}
        time.sleep(0.3)  # Rate limit
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code != 200:
            return None
        
        data = r.json()
        facts = data.get('facts', {})
        
        # Look for segment revenue in us-gaap or dei
        segment_mix = {}
        
        # Try to find revenue by segment
        # Common tags: us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
        # or segment-specific revenue disclosures
        us_gaap = facts.get('us-gaap', {})
        for tag, facts_list in us_gaap.items():
            if 'revenue' in tag.lower() and 'segment' in tag.lower():
                # Extract segment revenue
                for fact in facts_list.get('units', {}).get('USD', []):
                    segment = fact.get('segment', {}).get('dimension', {}).get('us-gaap:OperatingSegmentAxis', '')
                    if segment and fact.get('val'):
                        segment_mix[segment] = float(fact['val'])
        
        # Normalize to percentages
        if segment_mix:
            total = sum(segment_mix.values())
            if total > 0:
                segment_mix = {k: v / total for k, v in segment_mix.items()}
                # Cache result
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump({'segment_mix': segment_mix}, f)
                return segment_mix
        
        return None
    except Exception as e:
        print(f"  Error fetching XBRL for {ticker}: {e}")
        return None


if __name__ == "__main__":
    # Test
    segment_mix = fetch_xbrl_segment_revenue('AAPL', '0000320193')
    if segment_mix:
        print(f"Segment mix: {segment_mix}")
    else:
        print("No segment mix found")
