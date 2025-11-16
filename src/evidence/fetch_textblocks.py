"""
fetch_textblocks.py: Fetch SEC 10-K text using TextBlocks API.

API: https://api.textblocks.app
"""
import os
import requests
import time
from typing import Optional, Dict, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

TEXTBLOCKS_API_KEY = os.getenv("TEXTBLOCKS_API_KEY", "")
TEXTBLOCKS_EMAIL = os.getenv("TEXTBLOCKS_EMAIL", "")


def get_latest_10k_url(ticker: str, email: str = None, api_key: str = None) -> Optional[Dict]:
    """
    Get latest 10-K filing URL for a ticker using TextBlocks API.
    
    Returns dict with:
        - url: primaryDocumentUrl
        - filingDate: filing date
        - form: form type (should be "10-K")
    """
    email = email or TEXTBLOCKS_EMAIL
    api_key = api_key or TEXTBLOCKS_API_KEY
    
    if not email or not api_key:
        print(f"  Warning: TextBlocks API credentials not set")
        return None
    
    try:
        url = f"https://api.textblocks.app/filings"
        params = {
            "ticker": ticker,
            "email": email,
            "api_key": api_key
        }
        
        time.sleep(0.5)  # Rate limit
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"  Error fetching filings for {ticker}: HTTP {response.status_code}")
            return None
        
        filings = response.json()
        if not filings or not isinstance(filings, list):
            return None
        
        # Find latest 10-K
        for filing in filings:
            if filing.get("form") == "10-K":
                return {
                    "url": filing.get("primaryDocumentUrl"),
                    "filingDate": filing.get("filingDate"),
                    "form": filing.get("form")
                }
        
        return None
    except Exception as e:
        print(f"  Error fetching 10-K URL for {ticker}: {e}")
        return None


def extract_10k_item(filing_url: str, item: str = "1", email: str = None, api_key: str = None) -> Optional[str]:
    """
    Extract specific item text from 10-K filing using TextBlocks API.
    
    Args:
        filing_url: URL to the 10-K filing
        item: Item number to extract ("1", "1A", "7", "7A", etc.)
        email: TextBlocks email
        api_key: TextBlocks API key
    
    Returns:
        Extracted text content, or None if failed
    """
    email = email or TEXTBLOCKS_EMAIL
    api_key = api_key or TEXTBLOCKS_API_KEY
    
    if not email or not api_key:
        return None
    
    try:
        # Use the same endpoint format as the official API
        url = "https://api.textblocks.app/extractor"
        params = {
            "url": filing_url,
            "item": item,
            "email": email,
            "api_key": api_key
        }
        
        time.sleep(0.5)  # Rate limit
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"  Error extracting item {item}: HTTP {response.status_code}")
            return None
        
        data = response.json()
        # TextBlocks API returns text in different formats, check common fields
        text = data.get("text") or data.get("content") or data.get("item_text") or ""
        
        return text if text else None
    except Exception as e:
        print(f"  Error extracting item {item} from {filing_url}: {e}")
        return None


def fetch_10k_textblocks(ticker: str, items: List[str] = ["1"], email: str = None, api_key: str = None) -> Optional[Dict]:
    """
    Fetch 10-K text for a ticker using TextBlocks API.
    
    Args:
        ticker: Company ticker
        items: List of items to extract (e.g., ["1", "1A", "7"])
        email: TextBlocks email
        api_key: TextBlocks API key
    
    Returns:
        Dict with:
            - url: filing URL
            - filingDate: filing date
            - items: dict of {item: text} for each requested item
    """
    # Get latest 10-K URL
    filing_info = get_latest_10k_url(ticker, email, api_key)
    if not filing_info:
        return None
    
    filing_url = filing_info["url"]
    filing_date = filing_info.get("filingDate")
    
    # Extract requested items
    items_text = {}
    for item in items:
        text = extract_10k_item(filing_url, item, email, api_key)
        if text:
            items_text[item] = text
    
    if not items_text:
        return None
    
    return {
        "url": filing_url,
        "filingDate": filing_date,
        "type": "10K",
        "items": items_text
    }


if __name__ == "__main__":
    # Test
    result = fetch_10k_textblocks("HURN", items=["1"], email=TEXTBLOCKS_EMAIL, api_key=TEXTBLOCKS_API_KEY)
    if result:
        print(f"Fetched 10-K: {result['url']}")
        print(f"Item 1 length: {len(result['items'].get('1', ''))}")
    else:
        print("Failed to fetch 10-K (check API credentials)")

