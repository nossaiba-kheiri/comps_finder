"""
segment_vocab_cache.py: Cache segment vocabulary for reuse across pipeline runs.

Caches vocabulary per target and per company set to avoid rebuilding.
"""
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta


# Cache directory (relative to comps/ root)
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration (days)
CACHE_EXPIRY_DAYS = 30


def _get_company_fingerprint(company_data: Dict) -> str:
    """
    Generate a fingerprint for a company based on its segment data.
    
    Args:
        company_data: Company dict with segment_mix, customer_segment, etc.
    
    Returns:
        Hash string identifying this company's segment profile
    """
    # Extract segment-related fields
    segment_mix = company_data.get('segment_mix', {})
    customer_segment = company_data.get('customer_segment', [])
    
    # Normalize and sort for consistent hashing
    seg_mix_str = json.dumps(segment_mix, sort_keys=True) if segment_mix else ''
    seg_list_str = json.dumps(sorted(customer_segment), sort_keys=True) if customer_segment else ''
    
    # Combine and hash
    combined = f"{seg_mix_str}|{seg_list_str}"
    return hashlib.md5(combined.encode()).hexdigest()


def _get_vocab_cache_key(target_id: str, company_fingerprints: List[str]) -> str:
    """
    Generate cache key for vocabulary.
    
    Args:
        target_id: Target company identifier (e.g., 'huron_consulting_group_inc')
        company_fingerprints: List of company fingerprints (sorted for consistency)
    
    Returns:
        Cache key string
    """
    # Sort fingerprints for consistency
    sorted_fps = sorted(company_fingerprints)
    combined = f"{target_id}|{json.dumps(sorted_fps, sort_keys=True)}"
    return hashlib.md5(combined.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get file path for cache entry."""
    return CACHE_DIR / f"segment_vocab_{cache_key}.json"


def load_cached_vocabulary(
    target_id: str,
    all_companies: List[Dict]
) -> Optional[List[str]]:
    """
    Load cached segment vocabulary if available and valid.
    
    Args:
        target_id: Target company identifier
        all_companies: List of company dicts (target + candidates)
    
    Returns:
        Cached vocabulary list if found and valid, None otherwise
    """
    try:
        # Generate fingerprints for all companies
        company_fingerprints = [_get_company_fingerprint(c) for c in all_companies]
        
        # Generate cache key
        cache_key = _get_vocab_cache_key(target_id, company_fingerprints)
        cache_path = _get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Load cache entry
        with open(cache_path, 'r') as f:
            cache_entry = json.load(f)
        
        # Check expiration
        cached_time = datetime.fromisoformat(cache_entry.get('cached_at', ''))
        if datetime.now() - cached_time > timedelta(days=CACHE_EXPIRY_DAYS):
            # Expired - delete and return None
            cache_path.unlink()
            return None
        
        # Verify vocabulary matches (sanity check)
        vocabulary = cache_entry.get('vocabulary', [])
        if not vocabulary or not isinstance(vocabulary, list):
            return None
        
        return vocabulary
        
    except Exception as e:
        # If anything fails, return None (cache miss)
        return None


def save_cached_vocabulary(
    target_id: str,
    all_companies: List[Dict],
    vocabulary: List[str]
) -> None:
    """
    Save segment vocabulary to cache.
    
    Args:
        target_id: Target company identifier
        all_companies: List of company dicts (target + candidates)
        vocabulary: Vocabulary list to cache
    """
    try:
        # Generate fingerprints for all companies
        company_fingerprints = [_get_company_fingerprint(c) for c in all_companies]
        
        # Generate cache key
        cache_key = _get_vocab_cache_key(target_id, company_fingerprints)
        cache_path = _get_cache_path(cache_key)
        
        # Create cache entry
        cache_entry = {
            'target_id': target_id,
            'vocabulary': vocabulary,
            'num_companies': len(all_companies),
            'company_fingerprints': company_fingerprints,
            'cached_at': datetime.now().isoformat(),
            'cache_version': '1.0'
        }
        
        # Save to file
        with open(cache_path, 'w') as f:
            json.dump(cache_entry, f, indent=2)
        
    except Exception as e:
        # If caching fails, just continue (non-critical)
        pass


def clear_expired_cache() -> int:
    """
    Clear expired cache entries.
    
    Returns:
        Number of entries cleared
    """
    cleared = 0
    try:
        for cache_file in CACHE_DIR.glob('segment_vocab_*.json'):
            try:
                with open(cache_file, 'r') as f:
                    cache_entry = json.load(f)
                
                cached_time = datetime.fromisoformat(cache_entry.get('cached_at', ''))
                if datetime.now() - cached_time > timedelta(days=CACHE_EXPIRY_DAYS):
                    cache_file.unlink()
                    cleared += 1
            except Exception:
                # If file is corrupted, delete it
                cache_file.unlink()
                cleared += 1
    except Exception:
        pass
    
    return cleared

