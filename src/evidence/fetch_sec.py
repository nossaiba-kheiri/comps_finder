"""
fetch_sec.py: Download latest SEC 10-K Item 1 Business for a given ticker.
"""
import os
import requests
import time
import re
from bs4 import BeautifulSoup
from datetime import datetime

SEC_EDGAR_API = "https://data.sec.gov/submissions"
SEC_EDGAR_FILINGS = "https://www.sec.gov/cgi-bin/viewer"
CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../data/cache/sec')


def get_latest_10k_url(cik, ticker):
    """
    Get the latest 10-K filing URL for a company.
    Returns (filing_url, filing_date) or (None, None) if not found.
    """
    try:
        # Clean CIK: ensure it's 10 digits with leading zeros
        cik_clean = str(cik).lstrip('0')
        if not cik_clean:
            return None, None
        cik_padded = cik_clean.zfill(10)  # Pad to 10 digits
        
        # Get company submissions - SEC API requires CIK in format CIK0000320193
        url = f"{SEC_EDGAR_API}/CIK{cik_padded}.json"
        headers = {'User-Agent': 'CompFinder/1.0 you@email.com'}
        time.sleep(0.3)  # Rate limit
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"  SEC API returned status {r.status_code} for CIK {cik_padded}")
            return None, None
        
        data = r.json()
        filings = data.get('filings', {}).get('recent', {})
        forms = filings.get('form', [])
        filing_dates = filings.get('filingDate', [])
        accession_numbers = filings.get('accessionNumber', [])
        primary_documents = filings.get('primaryDocument', [])
        
        # Find latest 10-K
        for i, form in enumerate(forms):
            if form == '10-K':
                accession = accession_numbers[i].replace('-', '')
                filing_date = filing_dates[i]
                primary_doc = primary_documents[i]
                # Construct URL
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
                return filing_url, filing_date
        
        return None, None
    except Exception as e:
        print(f"  Error fetching 10-K URL for {ticker}: {e}")
        return None, None


def extract_item_from_html(html_content, item):
    """
    Extract a specific item from 10-K HTML.
    
    Args:
        html_content: HTML content of 10-K filing
        item: Item number to extract ("1", "1A", "7", "7A", etc.)
    
    Returns:
        Text content of the item, or empty string if not found
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Map item numbers to section names
        item_patterns = {
            "1": r'(?i)item\s+1[.\s]+business',
            "1A": r'(?i)item\s+1A[.\s]+(risk\s+factors|risks)',
            "7": r'(?i)item\s+7[.\s]+(management[.\s]+discussion|md&a)',
            "7A": r'(?i)item\s+7A[.\s]+(quantitative\s+and\s+qualitative|disclosures)',
        }
        
        # Default pattern if item not in map
        if item not in item_patterns:
            pattern = rf'(?i)item\s+{re.escape(item)}[.\s]+'
        else:
            pattern = item_patterns[item]
        
        match = re.search(pattern, text)
        if not match:
            return ""
        
        start_idx = match.start()
        
        # Find next major item
        # For Item 1, look for 1A or 2
        # For Item 1A, look for 2
        # For Item 7, look for 7A or 8
        # For Item 7A, look for 8
        next_items_map = {
            "1": ["1A", "2"],
            "1A": ["2"],
            "7": ["7A", "8"],
            "7A": ["8"]
        }
        
        next_items = next_items_map.get(item, [str(int(item) + 1) if item.isdigit() else ""])
        next_pattern = rf'(?i)item\s+({"|".join(next_items)})[.\s]+'
        
        next_match = re.search(next_pattern, text[start_idx + 100:])
        if next_match:
            end_idx = start_idx + 100 + next_match.start()
        else:
            # If no next item found, take next 50k characters
            end_idx = min(start_idx + 50000, len(text))
        
        item_text = text[start_idx:end_idx]
        # Clean up: remove excessive whitespace
        item_text = re.sub(r'\s+', ' ', item_text).strip()
        
        return item_text
    except Exception as e:
        print(f"  Error extracting Item {item}: {e}")
        return ""


def extract_item1_business(html_content):
    """
    Extract Item 1 (Business) section from 10-K HTML.
    Returns text content of Item 1.
    """
    return extract_item_from_html(html_content, "1")


def fetch_sec_10k(ticker, cik, items=None):
    """
    Fetch latest 10-K items for a ticker using direct SEC.gov access.
    
    Args:
        ticker: Company ticker
        cik: Company CIK
        items: List of items to extract (e.g., ["1", "1A", "7"]). Default: ["1"]
    
    Returns:
        Dict with url, filing_date, items (dict of {item: text}), or None if not found.
    """
    if not cik or not ticker:
        return None
    
    if items is None:
        items = ["1"]
    
    # Clean CIK (remove leading zeros if needed, but keep as string for URL)
    cik_str = str(cik).zfill(10)  # Pad to 10 digits with leading zeros
    cik_clean = str(cik).lstrip('0')
    if not cik_clean:
        return None
    
    try:
        # Get latest 10-K URL
        filing_url, filing_date = get_latest_10k_url(cik_clean, ticker)
        if not filing_url:
            return None
        
        # Check cache
        cache_file = os.path.join(CACHE_DIR, f"{ticker}_10k.json")
        cached_items = {}
        if os.path.exists(cache_file):
            import json
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                # Check if cache is fresh (same filing date)
                if cached.get('filing_date') == filing_date and 'items' in cached:
                    cached_items = cached.get('items', {})
        
        # Fetch 10-K HTML if we need items not in cache
        items_to_fetch = [item for item in items if item not in cached_items]
        if items_to_fetch:
            headers = {'User-Agent': 'CompFinder/1.0 you@email.com'}
            time.sleep(0.3)  # Rate limit
            r = requests.get(filing_url, headers=headers, timeout=30)
            if r.status_code != 200:
                return None
            
            # Extract requested items
            for item in items_to_fetch:
                item_text = extract_item_from_html(r.text, item)
                if item_text:
                    cached_items[item] = item_text[:20000]  # Limit to 20k chars
        
        if not cached_items:
            return None
        
        result = {
            'url': filing_url,
            'filing_date': filing_date,
            'type': '10K',
            'items': cached_items
        }
        
        # Cache result
        os.makedirs(CACHE_DIR, exist_ok=True)
        import json
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        
        return result
    except Exception as e:
        print(f"  Error fetching 10-K for {ticker}: {e}")
        return None


if __name__ == "__main__":
    # Test
    result = fetch_sec_10k('AAPL', '0000320193')
    if result:
        print(f"Fetched 10-K: {result['url']}")
        print(f"Text length: {len(result['text'])}")
    else:
        print("Failed to fetch 10-K")
