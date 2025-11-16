"""
fetch_site.py: Fetch pages (/about, /services, /solutions, /industries) for a company website.
"""
import os
import requests
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../data/cache/site')
MAX_PAGES = 5
MAX_SIZE = 1.5 * 1024 * 1024  # 1.5 MB


def normalize_url(base_url, path):
    """Normalize URL, handle relative paths."""
    if not base_url:
        return None
    if not base_url.startswith('http'):
        base_url = 'https://' + base_url
    return urljoin(base_url, path)


def extract_text_from_html(html_content):
    """Extract clean text from HTML."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except Exception:
        return ""


def fetch_site_evidence(website_url, config=None):
    """
    Fetch evidence from company website pages.
    Returns list of dicts with {type, url, text} for each page.
    """
    # Handle NaN, None, or empty values
    if not website_url or pd.isna(website_url) or (isinstance(website_url, float) and np.isnan(website_url)):
        return []
    
    # Convert to string if not already
    if not isinstance(website_url, str):
        website_url = str(website_url).strip()
    
    config = config or {}
    pages_to_fetch = config.get('pages', ['/about', '/services', '/solutions', '/industries'])
    max_pages = config.get('max_pages', MAX_PAGES)
    max_size = config.get('max_size', MAX_SIZE)
    
    # Normalize base URL
    base_url = website_url.strip()
    if not base_url or base_url == 'nan' or base_url == 'None':
        return []
    
    if not base_url.startswith('http'):
        base_url = 'https://' + base_url
    
    parsed = urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    
    # Check cache
    cache_key = parsed.netloc.replace('.', '_')
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        import json
        from datetime import datetime, timedelta
        try:
            # Check cache age (4 months as per requirements)
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(days=120):  # 4 months
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    sources = cached.get('sources', [])
                    if sources:
                        # Return cached sources with a flag indicating cache hit
                        # We'll check this in pack.py to print appropriate message
                        return sources  # Return cached data
        except Exception as e:
            # If cache read fails, continue with fresh fetch
            pass
    
    sources = []
    total_size = 0
    
    for page_path in pages_to_fetch[:max_pages]:
        if total_size >= max_size:
            break
        
        try:
            page_url = normalize_url(domain, page_path)
            if not page_url:
                continue
            
            # Fetch page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            time.sleep(0.5)  # Rate limit
            r = requests.get(page_url, headers=headers, timeout=10, allow_redirects=True)
            
            if r.status_code == 200:
                text = extract_text_from_html(r.text)
                if text:
                    size = len(text.encode('utf-8'))
                    total_size += size
                    sources.append({
                        'type': 'site',
                        'url': page_url,
                        'text': text[:10000],  # Limit to 10k chars per page
                        'section': page_path
                    })
        except Exception as e:
            # Silently skip failed pages
            continue
    
    # Cache results
    if sources:
        os.makedirs(CACHE_DIR, exist_ok=True)
        import json
        from datetime import datetime
        with open(cache_file, 'w') as f:
            json.dump({
                'sources': sources,
                'domain': domain,
                'cached_at': datetime.now().isoformat(),
                'url': base_url
            }, f)
    
    return sources


if __name__ == "__main__":
    # Test
    sources = fetch_site_evidence('https://www.apple.com')
    print(f"Fetched {len(sources)} pages")
    for source in sources:
        print(f"  {source['url']}: {len(source['text'])} chars")
