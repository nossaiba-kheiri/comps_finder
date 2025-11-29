"""
llm_extract_cache.py: Cache LLM extraction results to avoid redundant API calls.
"""
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "llm_extraction"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRATION_DAYS = 30


def _hash_evidence_pack(evidence_pack: Dict) -> str:
    """
    Generate a hash of the evidence pack for cache key.
    
    Args:
        evidence_pack: EvidencePack dict with sources
    
    Returns:
        str: Hex digest of the hash
    """
    # Extract key fields that determine extraction result
    # We hash the text content, not the full structure
    text_parts = []
    
    # Extract text from sources
    for s in evidence_pack.get('sources', []):
        if s.get('type') == '10K' and 'items' in s:
            # 10-K items
            items = s.get('items', {})
            for item_text in items.values():
                text_parts.append(str(item_text)[:5000])  # Truncate for hash
        elif 'text' in s:
            # Website text
            text_parts.append(str(s.get('text', ''))[:5000])  # Truncate for hash
    
    # Also include segment_mix_xbrl if present
    if evidence_pack.get('segment_mix_xbrl'):
        text_parts.append(json.dumps(evidence_pack.get('segment_mix_xbrl'), sort_keys=True))
    
    # Combine and hash
    combined = '|'.join(text_parts)
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    return hash_obj.hexdigest()


def _get_cache_key(ticker: str, evidence_pack: Dict, prompt_version: str = 'svc_cust_v3') -> str:
    """
    Generate a unique cache key based on ticker, evidence hash, and prompt version.
    
    Args:
        ticker: Company ticker symbol
        evidence_pack: EvidencePack dict
        prompt_version: Prompt version string
    
    Returns:
        str: Cache key (filename-safe)
    """
    evidence_hash = _hash_evidence_pack(evidence_pack)
    safe_ticker = ticker.replace('/', '_').replace('\\', '_').replace(' ', '_').upper()
    return f"{safe_ticker}_{evidence_hash[:16]}_{prompt_version}"


def load_cached_extraction(
    ticker: str,
    evidence_pack: Dict,
    prompt_version: str = 'svc_cust_v3'
) -> Optional[Dict]:
    """
    Load cached LLM extraction result if available and not expired.
    
    Args:
        ticker: Company ticker symbol
        evidence_pack: EvidencePack dict
        prompt_version: Prompt version string
    
    Returns:
        Dict with extracted data or None if not found/expired
    """
    cache_key = _get_cache_key(ticker, evidence_pack, prompt_version)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    metadata_file = CACHE_DIR / f"{cache_key}_metadata.json"
    
    if not cache_file.exists() or not metadata_file.exists():
        return None
    
    try:
        # Check expiration
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        created_at = datetime.fromisoformat(metadata.get('created_at', ''))
        if datetime.now() - created_at > timedelta(days=CACHE_EXPIRATION_DAYS):
            # Expired - delete cache files
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            return None
        
        # Verify evidence hash matches (safety check)
        cached_hash = metadata.get('evidence_hash')
        current_hash = _hash_evidence_pack(evidence_pack)
        if cached_hash != current_hash:
            # Evidence changed - invalidate cache
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            return None
        
        # Verify prompt version matches
        cached_prompt_version = metadata.get('prompt_version')
        if cached_prompt_version != prompt_version:
            # Prompt version changed - invalidate cache
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            return None
        
        # Load cached extraction
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        return cached_data.get('extracted')
    except Exception as e:
        # If loading fails, delete corrupted cache
        cache_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
        return None


def save_cached_extraction(
    ticker: str,
    evidence_pack: Dict,
    extracted: Dict,
    prompt_version: str = 'svc_cust_v3'
):
    """
    Save LLM extraction result to cache.
    
    Args:
        ticker: Company ticker symbol
        evidence_pack: EvidencePack dict
        extracted: Extracted data dict from LLM
        prompt_version: Prompt version string
    """
    cache_key = _get_cache_key(ticker, evidence_pack, prompt_version)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    metadata_file = CACHE_DIR / f"{cache_key}_metadata.json"
    
    try:
        # Save extraction result
        cached_data = {
            'extracted': extracted
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        # Save metadata
        evidence_hash = _hash_evidence_pack(evidence_pack)
        metadata = {
            'ticker': ticker,
            'evidence_hash': evidence_hash,
            'prompt_version': prompt_version,
            'created_at': datetime.now().isoformat(),
            'cache_key': cache_key
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        # Non-critical - continue even if caching fails
        pass


def clear_expired_extraction_cache() -> int:
    """
    Clear expired LLM extraction cache files.
    
    Returns:
        int: Number of cache files cleared
    """
    cleared = 0
    try:
        for cache_file in CACHE_DIR.glob("*.pkl"):
            # Find corresponding metadata file
            metadata_file = cache_file.parent / f"{cache_file.stem}_metadata.json"
            if not metadata_file.exists():
                # Orphaned cache file - delete it
                cache_file.unlink(missing_ok=True)
                cleared += 1
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                created_at = datetime.fromisoformat(metadata.get('created_at', ''))
                if datetime.now() - created_at > timedelta(days=CACHE_EXPIRATION_DAYS):
                    cache_file.unlink(missing_ok=True)
                    metadata_file.unlink(missing_ok=True)
                    cleared += 1
            except Exception:
                # Corrupted metadata - delete both files
                cache_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
                cleared += 1
    except Exception:
        pass
    
    return cleared

