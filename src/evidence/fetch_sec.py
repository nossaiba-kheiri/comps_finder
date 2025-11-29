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
        
        # Remove table of contents and navigation elements
        for toc in soup.find_all(['table', 'nav', 'div'], class_=re.compile(r'toc|contents|navigation', re.I)):
            toc.decompose()
        
        # Get text with better structure preservation
        text = soup.get_text(separator='\n', strip=True)
        
        # Map item numbers to section names
        item_patterns = {
            "1": r'(?i)^\s*item\s+1[.\s]*business\s*$',
            "1A": r'(?i)^\s*item\s+1A[.\s]*(risk\s+factors|risks)\s*$',
            "7": r'(?i)^\s*item\s+7[.\s]*(management[.\s]*discussion|md&a)\s*$',
            "7A": r'(?i)^\s*item\s+7A[.\s]*(quantitative\s+and\s+qualitative|disclosures)\s*$',
        }
        
        # Default pattern if item not in map
        if item not in item_patterns:
            pattern = rf'(?i)^\s*item\s+{re.escape(item)}[.\s]*'
        else:
            pattern = item_patterns[item]
        
        # Split into lines and find the actual section header (not TOC)
        lines = text.split('\n')
        start_idx = None
        toc_indicators = ['table of contents', 'contents', 'page', 'item 1a', 'item 2']
        
        for i, line in enumerate(lines):
            # Skip lines that look like TOC entries (short lines with page numbers)
            if len(line.strip()) < 50 and any(indicator in line.lower() for indicator in toc_indicators):
                continue
            
            # Look for the actual section header
            if re.search(pattern, line):
                # Make sure it's not just a TOC entry - check if next lines have substantial content
                if i + 1 < len(lines) and len(lines[i + 1].strip()) > 100:
                    start_idx = i
                    break
        
        if start_idx is None:
            # Fallback: search in full text (old method)
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                # Find the start of the line containing the match
                text_before = text[:match.start()]
                start_idx = len(text_before.split('\n'))
            else:
                return ""
        
        # Reconstruct text from start_idx
        item_lines = lines[start_idx:]
        item_text = '\n'.join(item_lines)
        
        # Find next major item
        next_items_map = {
            "1": ["1A", "2"],
            "1A": ["2"],
            "7": ["7A", "8"],
            "7A": ["8"]
        }
        
        next_items = next_items_map.get(item, [str(int(item) + 1) if item.isdigit() else ""])
        next_patterns = [rf'(?i)^\s*item\s+{re.escape(ni)}[.\s]*' for ni in next_items]
        
        # Find the next item in the remaining lines
        end_idx = len(item_lines)
        for i, line in enumerate(item_lines[100:], start=100):  # Skip first 100 lines to avoid TOC
            for next_pattern in next_patterns:
                if re.search(next_pattern, line):
                    # Make sure it's a real section, not TOC
                    if len(line.strip()) > 50 or (i + 1 < len(item_lines) and len(item_lines[i + 1].strip()) > 100):
                        end_idx = i
                        break
            if end_idx < len(item_lines):
                break
        
        item_text = '\n'.join(item_lines[:end_idx])
        
        # Clean up: remove excessive whitespace but preserve paragraph structure
        item_text = re.sub(r'[ \t]+', ' ', item_text)  # Collapse spaces/tabs
        item_text = re.sub(r'\n{3,}', '\n\n', item_text)  # Max 2 newlines
        item_text = item_text.strip()
        
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
