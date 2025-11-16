"""
fetch_linkedin.py: Fetch LinkedIn company posts using Apify LinkedIn scraper.

API: https://apify.com/apimaestro/linkedin-company-posts-batch-scraper-no-cookies

Caching: LinkedIn posts are cached for 4 months to avoid redundant fetches.
"""
import os
import json
import hashlib
from datetime import datetime, timedelta, timezone
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

# Cache directory for LinkedIn results
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "linkedin"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration: 4 months
CACHE_EXPIRATION_DAYS = 4 * 30  # 120 days


def _normalize_linkedin_identifier(company_name: str = None, company_url: str = None) -> str:
    """Normalize LinkedIn company identifier for cache key generation."""
    if company_url:
        # Extract company name from URL if possible
        if "linkedin.com/company/" in company_url:
            parts = company_url.split("linkedin.com/company/")
            if len(parts) > 1:
                company_id = parts[1].split("/")[0].split("?")[0].strip().lower()
                return company_id
        # Otherwise use full URL (normalized)
        url = company_url.rstrip('/').lower()
        if url.startswith('http://'):
            url = url[7:]
        elif url.startswith('https://'):
            url = url[8:]
        if url.startswith('www.'):
            url = url[4:]
        return url
    elif company_name:
        return company_name.strip().lower()
    else:
        return "unknown"


def _get_linkedin_cache_key(company_name: str = None, company_url: str = None, months_back: int = 8) -> str:
    """Generate cache key from LinkedIn company identifier and months_back."""
    # Normalize identifier
    company_id = _normalize_linkedin_identifier(company_name, company_url)
    
    # Create a hash from company identifier and months_back
    params_str = json.dumps({
        "company_id": company_id,
        "months_back": months_back
    }, sort_keys=True)
    
    # Generate hash
    cache_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    # Create safe filename from company identifier
    safe_id = company_id.replace('/', '_').replace('.', '_').replace(':', '_').replace('-', '_')
    # Limit length
    if len(safe_id) > 100:
        safe_id = safe_id[:100]
    
    return f"{safe_id}_{cache_hash}"


def _get_linkedin_cache_path(cache_key: str) -> Path:
    """Get cache file path for a cache key."""
    return CACHE_DIR / f"{cache_key}.json"


def _load_linkedin_from_cache(cache_path: Path) -> Optional[List[Dict]]:
    """Load cached LinkedIn posts if they exist and are not expired."""
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
        print(f"    Using cached LinkedIn posts (cached {age_days} days ago)")
        return cache_data.get('posts')
    
    except Exception as e:
        print(f"    Warning: Error loading cache: {e}")
        # Delete corrupted cache
        if cache_path.exists():
            cache_path.unlink()
        return None


def _save_linkedin_to_cache(cache_path: Path, posts: List[Dict], company_name: str = None, company_url: str = None):
    """Save fetched LinkedIn posts to cache."""
    try:
        cache_data = {
            'company_name': company_name,
            'company_url': company_url,
            'cached_at': datetime.now().isoformat(),
            'cached_at_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'posts_count': len(posts),
            'posts': posts
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"    Saved {len(posts)} LinkedIn posts to cache")
    
    except Exception as e:
        print(f"    Warning: Error saving cache: {e}")


def fetch_linkedin_posts(
    company_name: str = None,
    company_url: str = None,
    months_back: int = 8,
    api_token: str = None,
    use_cache: bool = True,
    force_refresh: bool = False
) -> List[Dict]:
    """
    Fetch LinkedIn company posts from last N months.
    
    Caching: Results are cached for 4 months. Set force_refresh=True to bypass cache.
    
    Args:
        company_name: LinkedIn company vanity name (e.g., "huron-consulting-group")
        company_url: Full LinkedIn company URL (e.g., "https://www.linkedin.com/company/huron-consulting-group/")
        months_back: Number of months to look back (default: 8)
        api_token: Apify API token (or use APIFY_API_TOKEN env var)
        use_cache: Whether to use cached results (default: True)
        force_refresh: Whether to force a fresh fetch even if cache exists (default: False)
    
    Returns:
        List of dicts with:
            - postUrl: post URL
            - content: post text
            - postedAt: ISO timestamp
            - stats: dict with likes, comments, reposts
    """
    # Check cache first (unless force_refresh)
    if use_cache and not force_refresh:
        cache_key = _get_linkedin_cache_key(company_name, company_url, months_back)
        cache_path = _get_linkedin_cache_path(cache_key)
        cached_posts = _load_linkedin_from_cache(cache_path)
        
        if cached_posts is not None:
            return cached_posts
    
    api_token = api_token or APIFY_API_TOKEN
    
    if not APIFY_AVAILABLE:
        print("  Warning: apify_client not installed. Install with: pip install apify-client")
        return []
    
    if not api_token:
        print("  Warning: APIFY_API_TOKEN not set")
        return []
    
    if not company_name and not company_url:
        print("  Warning: Must provide company_name or company_url")
        return []
    
    try:
        client = ApifyClient(api_token)
        
        # Build company_names list
        company_names = []
        if company_name:
            company_names.append(company_name)
        if company_url:
            company_names.append(company_url)
        
        # Optimize for memory efficiency
        run_input = {
            "company_names": company_names,
            "limit": 20,  # Limit to 20 posts max to reduce memory
            "maxRetries": 1  # Reduce retries to save memory
        }
        
        print(f"    Fetching LinkedIn posts for {company_name or company_url}...")
        run = client.actor(
            "apimaestro/linkedin-company-posts-batch-scraper-no-cookies"
        ).call(run_input=run_input)
        
        # Check if dataset exists and has items
        dataset = client.dataset(run["defaultDatasetId"])
        dataset_items = list(dataset.iterate_items())
        
        # If no items returned, the company page may not exist or have no posts
        if len(dataset_items) == 0:
            print(f"    ⚠ No LinkedIn posts found (company page may not exist or have no posts)")
            return []
        
        # Get results and filter by date
        cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
        posts = []
        all_items = []
        date_parse_errors = 0
        too_old_count = 0
        no_date_count = 0
        
        for item in dataset_items:
            all_items.append(item)
            # Try multiple possible date field names
            posted_at_raw = item.get("postedAt") or item.get("posted_at") or item.get("createdAt") or item.get("created_at") or item.get("date")
            
            # Extract date from dictionary if needed
            # Apify returns posted_at as a dict: {'date': '2025-11-14 15:55:09', 'relative': '20h', ...}
            posted_at = None
            if isinstance(posted_at_raw, dict):
                # Extract from dict: try 'date' field first, then 'timestamp'
                posted_at = posted_at_raw.get("date") or posted_at_raw.get("timestamp")
                # If timestamp, convert to ISO string
                if posted_at and isinstance(posted_at, (int, float)):
                    posted_at = datetime.fromtimestamp(posted_at / 1000, tz=timezone.utc).isoformat()
            elif posted_at_raw:
                posted_at = posted_at_raw
            
            if not posted_at:
                no_date_count += 1
                # If no date, include it anyway (let downstream filter handle it)
                posts.append({
                    "postUrl": item.get("postUrl") or item.get("post_url", ""),
                    "content": item.get("content") or item.get("text", ""),
                    "postedAt": "",
                    "stats": item.get("stats", {})
                })
                continue
            
            # Parse timestamp - try multiple formats
            dt = None
            parsed = False
            
            if isinstance(posted_at, str):
                # Try various date formats
                try:
                    # Format 1: ISO with Z (most common)
                    if "Z" in posted_at:
                        dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
                        parsed = True
                    # Format 2: ISO with timezone offset
                    elif "+" in posted_at or "-" in posted_at[-6:]:
                        dt = datetime.fromisoformat(posted_at)
                        parsed = True
                    # Format 3: ISO without timezone (assume UTC)
                    elif "T" in posted_at:
                        dt = datetime.fromisoformat(posted_at)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        parsed = True
                    # Format 4: Date and time with space (YYYY-MM-DD HH:MM:SS) - Apify format!
                    elif " " in posted_at and len(posted_at) > 10:
                        # Format: '2025-11-14 15:55:09'
                        dt = datetime.strptime(posted_at, "%Y-%m-%d %H:%M:%S")
                        dt = dt.replace(tzinfo=timezone.utc)
                        parsed = True
                    # Format 5: Date only (YYYY-MM-DD)
                    elif len(posted_at) == 10 and posted_at.count("-") == 2:
                        dt = datetime.fromisoformat(posted_at + "T00:00:00+00:00")
                        parsed = True
                except Exception:
                    pass
            
            if not parsed:
                date_parse_errors += 1
                # DEBUG: Print the actual date value we couldn't parse
                if len(all_items) <= 3:  # Only print for first few items
                    print(f"    Debug: Could not parse date: {repr(posted_at)} (type: {type(posted_at)})")
                # Include posts with date parsing errors (might be valid but different format)
                posts.append({
                    "postUrl": item.get("postUrl") or item.get("post_url", ""),
                    "content": item.get("content") or item.get("text", ""),
                    "postedAt": str(posted_at) if posted_at else "",
                    "stats": item.get("stats", {})
                })
                continue
            
            # Filter by date - but be lenient: if date parsing succeeded, include if >= cutoff OR within last 12 months
            # This handles cases where system date might be wrong or posts are slightly older
            lenient_cutoff = cutoff - timedelta(days=60)  # Allow 2 more months of buffer
            if dt >= lenient_cutoff:
                posts.append({
                    "postUrl": item.get("postUrl") or item.get("post_url", ""),
                    "content": item.get("content") or item.get("text", ""),
                    "postedAt": posted_at,  # Use extracted date string
                    "stats": item.get("stats", {})
                })
            else:
                too_old_count += 1
        
        # Debug output - always show details if filtering happened
        if len(posts) == 0 and len(all_items) > 0:
            # Show sample dates from first few items
            sample_dates = []
            sample_items_debug = []
            for item in all_items[:5]:
                posted_at_raw = item.get("postedAt") or item.get("posted_at") or item.get("createdAt")
                if posted_at_raw:
                    # Extract from dict if needed
                    if isinstance(posted_at_raw, dict):
                        posted_at_str = posted_at_raw.get("date") or str(posted_at_raw.get("timestamp", ""))
                    else:
                        posted_at_str = posted_at_raw
                    sample_dates.append(posted_at_str)
                    # Try to parse it to show what it looks like
                    try:
                        if isinstance(posted_at_str, str):
                            if " " in posted_at_str and len(posted_at_str) > 10:
                                # Apify format: '2025-11-14 15:55:09'
                                dt_parsed = datetime.strptime(posted_at_str, "%Y-%m-%d %H:%M:%S")
                                dt_parsed = dt_parsed.replace(tzinfo=timezone.utc)
                            elif "Z" in posted_at_str:
                                dt_parsed = datetime.fromisoformat(posted_at_str.replace("Z", "+00:00"))
                            else:
                                dt_parsed = datetime.fromisoformat(posted_at_str)
                            sample_items_debug.append(f"{posted_at_str} → {dt_parsed.date()}")
                    except Exception as e:
                        sample_items_debug.append(f"{posted_at_str} → PARSE FAILED: {e}")
            
            print(f"    ⚠ DEBUG: Date filtering issue detected!")
            print(f"    Debug: Current date: {datetime.now(timezone.utc).isoformat()}")
            print(f"    Debug: Cutoff date ({months_back} months): {cutoff.isoformat()}")
            print(f"    Debug: Lenient cutoff: {lenient_cutoff.isoformat()}")
            print(f"    Debug: Total items from Apify: {len(all_items)}")
            print(f"    Debug: Sample post dates from Apify:")
            for sample in sample_items_debug[:3]:
                print(f"      - {sample}")
            print(f"    Debug: {too_old_count} posts were too old (before lenient cutoff)")
            print(f"    Debug: {date_parse_errors} posts had date parsing errors")
            print(f"    Debug: {no_date_count} posts had no date field")
            print(f"    Debug: {len(posts)} posts passed filter")
            
            # If all posts were filtered, show first item structure for debugging
            if len(all_items) > 0:
                first_item = all_items[0]
                print(f"    Debug: First item keys: {list(first_item.keys())}")
                print(f"    Debug: First item date fields:")
                for key in ['postedAt', 'posted_at', 'createdAt', 'created_at', 'date', 'timestamp']:
                    if key in first_item:
                        print(f"      {key}: {repr(first_item[key])}")
        elif len(posts) < len(all_items):
            print(f"    Note: {len(all_items)} posts fetched, {len(posts)} passed date filter ({too_old_count} too old)")
        
        print(f"    Found {len(posts)} posts from last {months_back} months (including posts without dates)")
        
        # Save to cache (even if empty - to cache "not found" result)
        if use_cache:
            cache_key = _get_linkedin_cache_key(company_name, company_url, months_back)
            cache_path = _get_linkedin_cache_path(cache_key)
            _save_linkedin_to_cache(cache_path, posts, company_name, company_url)
        
        return posts
    
    except Exception as e:
        error_msg = str(e).lower()
        # Check for specific error types
        if "not found" in error_msg or "404" in error_msg:
            print(f"  ⚠ LinkedIn company page not found - may not exist")
        elif "timeout" in error_msg:
            print(f"  ⚠ Request timed out")
        elif "memory" in error_msg:
            print(f"  ⚠ Memory limit reached")
        else:
            print(f"  Error fetching LinkedIn posts: {e}")
        return []


def extract_company_name_from_url(url: str) -> Optional[str]:
    """Extract LinkedIn company vanity name from URL."""
    if not url:
        return None
    
    # Handle various LinkedIn URL formats
    if "linkedin.com/company/" in url:
        parts = url.split("linkedin.com/company/")
        if len(parts) > 1:
            company_part = parts[1].split("/")[0].split("?")[0]
            return company_part
    
    return None


if __name__ == "__main__":
    # Test
    posts = fetch_linkedin_posts(
        company_name="huron-consulting-group",
        months_back=8
    )
    print(f"\nFetched {len(posts)} posts")
    for post in posts[:3]:
        print(f"  {post['postedAt']}: {post['content'][:100]}...")

