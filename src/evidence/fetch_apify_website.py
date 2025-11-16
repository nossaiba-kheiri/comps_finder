"""
fetch_apify_website.py: Fetch website content using Apify Website Content Crawler.

API: https://apify.com/apify/website-content-crawler

Caching: Website content is cached for 4 months to avoid redundant crawls.
"""
import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

# Cache directory for website crawler results
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "website_crawler"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration: 4 months
CACHE_EXPIRATION_DAYS = 4 * 30  # 120 days


def _normalize_url(url: str) -> str:
    """Normalize URL for cache key generation."""
    # Remove trailing slash and convert to lowercase
    url = url.rstrip('/').lower()
    # Remove protocol
    if url.startswith('http://'):
        url = url[7:]
    elif url.startswith('https://'):
        url = url[8:]
    # Remove www.
    if url.startswith('www.'):
        url = url[4:]
    return url


def _get_cache_key(url: str, include_globs: List[str] = None, exclude_globs: List[str] = None, max_pages: int = 10) -> str:
    """Generate cache key from URL and crawl parameters."""
    # Normalize URL
    normalized_url = _normalize_url(url)
    
    # Create a hash from URL and parameters
    params_str = json.dumps({
        "url": normalized_url,
        "include_globs": sorted(include_globs) if include_globs else None,
        "exclude_globs": sorted(exclude_globs) if exclude_globs else None,
        "max_pages": max_pages
    }, sort_keys=True)
    
    # Generate hash
    cache_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    # Create safe filename from normalized URL
    safe_url = normalized_url.replace('/', '_').replace('.', '_').replace(':', '_')
    # Limit length
    if len(safe_url) > 100:
        safe_url = safe_url[:100]
    
    return f"{safe_url}_{cache_hash}"


def _get_cache_path(cache_key: str) -> Path:
    """Get cache file path for a cache key."""
    return CACHE_DIR / f"{cache_key}.json"


def _load_from_cache(cache_path: Path) -> Optional[List[Dict]]:
    """Load cached website content if it exists and is not expired."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check expiration
        cached_at_str = cache_data.get('cached_at')
        if not cached_at_str:
            return None  # Invalid cache (no timestamp)
        
        cached_at = datetime.fromisoformat(cached_at_str)
        age_days = (datetime.now() - cached_at).days
        
        if age_days > CACHE_EXPIRATION_DAYS:
            # Cache expired, delete it
            cache_path.unlink()
            print(f"    Cache expired ({age_days} days old, max {CACHE_EXPIRATION_DAYS} days)")
            return None
        
        # Cache is valid
        print(f"    Using cached website content (cached {age_days} days ago)")
        return cache_data.get('pages')
    
    except Exception as e:
        print(f"    Warning: Error loading cache: {e}")
        # Delete corrupted cache
        if cache_path.exists():
            cache_path.unlink()
        return None


def _save_to_cache(cache_path: Path, pages: List[Dict], url: str):
    """Save crawled website content to cache."""
    try:
        cache_data = {
            'url': url,
            'cached_at': datetime.now().isoformat(),
            'cached_at_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pages_count': len(pages),
            'pages': pages
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"    Saved {len(pages)} pages to cache")
    
    except Exception as e:
        print(f"    Warning: Error saving cache: {e}")


def fetch_website_content(
    url: str,
    include_globs: List[str] = None,
    exclude_globs: List[str] = None,
    max_pages: int = 10,  # Reduced default from 40 to 10 for memory efficiency
    api_token: str = None,
    use_cache: bool = True,
    force_refresh: bool = False
) -> List[Dict]:
    """
    Fetch website content using Apify Website Content Crawler.
    
    Caching: Results are cached for 4 months. Set force_refresh=True to bypass cache.
    
    Args:
        url: Base URL to crawl
        include_globs: URL glob patterns to include (e.g., ["**/services/**", "**/solutions/**"])
        exclude_globs: URL glob patterns to exclude (e.g., ["**/careers/**", "**/news/**"])
        max_pages: Maximum number of pages to crawl
        api_token: Apify API token (or use APIFY_API_TOKEN env var)
        use_cache: Whether to use cached results (default: True)
        force_refresh: Whether to force a fresh crawl even if cache exists (default: False)
    
    Returns:
        List of dicts with:
            - url: page URL
            - text: extracted text content
            - metadata: dict with title, description, etc.
    """
    # Check cache first (unless force_refresh)
    if use_cache and not force_refresh:
        cache_key = _get_cache_key(url, include_globs, exclude_globs, max_pages)
        cache_path = _get_cache_path(cache_key)
        cached_pages = _load_from_cache(cache_path)
        
        if cached_pages is not None:
            return cached_pages
        
        # If cache not found with exact parameters, try to find any cache for this URL
        # (fallback to use existing cache even if parameters differ slightly)
        if not cache_path.exists():
            # Try to find existing cache files for this URL
            normalized_url = _normalize_url(url)
            safe_url = normalized_url.replace('/', '_').replace('.', '_').replace(':', '_')
            if len(safe_url) > 100:
                safe_url = safe_url[:100]
            
            # Look for any cache file starting with this URL prefix
            cache_dir = cache_path.parent
            if cache_dir.exists():
                for cache_file in cache_dir.glob(f"{safe_url}_*.json"):
                    fallback_pages = _load_from_cache(cache_file)
                    if fallback_pages is not None:
                        print(f"    Using cached website content from different parameters (cache file: {cache_file.name})")
                        return fallback_pages
    
    api_token = api_token or APIFY_API_TOKEN
    
    if not APIFY_AVAILABLE:
        print("  Warning: apify_client not installed. Install with: pip install apify-client")
        return []
    
    if not api_token:
        print("  Warning: APIFY_API_TOKEN not set")
        return []
    
    try:
        client = ApifyClient(api_token)
        
        # Build include/exclude globs
        if include_globs is None:
            # Default: include services, solutions, industries pages
            base_domain = url.rstrip('/')
            include_globs = [
                f"{base_domain}/services/**",
                f"{base_domain}/solutions/**",
                f"{base_domain}/industries/**",
                f"{base_domain}/about/**"
            ]
        
        if exclude_globs is None:
            # Default: exclude careers, news, blog
            base_domain = url.rstrip('/')
            exclude_globs = [
                f"{base_domain}/careers/**",
                f"{base_domain}/news/**",
                f"{base_domain}/blog/**",
                f"{base_domain}/press/**"
            ]
        
        # Optimize for memory efficiency:
        # - Use HTTP crawler (lighter than Playwright)
        # - Limit pages and depth
        # - Block media and heavy content
        # - Remove unnecessary elements
        run_input = {
            "startUrls": [{"url": url}],
            "useSitemaps": False,
            "respectRobotsTxtFile": True,
            "crawlerType": "cheerio",  # Cheerio is lightest (no browser, just HTML parsing)
            "includeUrlGlobs": include_globs,
            "excludeUrlGlobs": exclude_globs,
            "maxCrawlPages": max_pages,
            "maxCrawlDepth": 2,  # Limit depth to reduce memory
            "keepElementsCssSelector": "",
            "removeElementsCssSelector": "nav, footer, script, style, noscript, svg, img, video, audio, iframe, embed, object",
            "blockMedia": True,
            "blockResources": True,  # Block images, fonts, etc. to save memory
            "clickElementsCssSelector": "",  # Disable auto-clicking to reduce memory
            "storeSkippedUrls": False,
            "maxRequestRetries": 1,  # Reduce retries to save memory
            "requestHandlerTimeoutSecs": 30,  # Timeout to prevent hanging
            "maxConcurrency": 1  # Single request at a time to reduce memory
        }
        
        print(f"    Crawling {url} (max {max_pages} pages)...")
        run = client.actor("apify/website-content-crawler").call(run_input=run_input)
        
        # Get results
        pages = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            pages.append({
                "url": item.get("url", ""),
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {})
            })
        
        print(f"    Crawled {len(pages)} pages")
        
        # Save to cache
        if use_cache:
            cache_key = _get_cache_key(url, include_globs, exclude_globs, max_pages)
            cache_path = _get_cache_path(cache_key)
            _save_to_cache(cache_path, pages, url)
        
        return pages
    
    except Exception as e:
        print(f"  Error crawling website {url}: {e}")
        return []


if __name__ == "__main__":
    # Test
    pages = fetch_website_content("https://www.huronconsultinggroup.com", max_pages=10)
    print(f"\nFetched {len(pages)} pages")
    for page in pages[:3]:
        print(f"  {page['url']}: {len(page['text'])} chars")

