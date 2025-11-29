"""
shap_cache.py: Cache SHAP model and values to avoid recomputation when dataset is identical.
"""
import json
import hashlib
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "shap"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRATION_DAYS = 30


def _hash_features_df(features_df, feature_cols=['P', 'C', 'S']):
    """
    Generate a hash of the feature matrix.
    
    Args:
        features_df: DataFrame with feature columns
        feature_cols: List of feature column names to hash
    
    Returns:
        str: Hex digest of the hash
    """
    # Extract feature columns and sort by index to ensure consistent ordering
    feature_data = features_df[feature_cols].sort_index().values
    
    # Convert to bytes and hash
    feature_bytes = feature_data.tobytes()
    hash_obj = hashlib.sha256(feature_bytes)
    
    # Also include shape and column names in hash for safety
    shape_str = f"{feature_data.shape[0]}_{feature_data.shape[1]}"
    cols_str = "_".join(sorted(feature_cols))
    hash_obj.update(shape_str.encode())
    hash_obj.update(cols_str.encode())
    
    return hash_obj.hexdigest()


def _get_cache_key(target_id: str, features_df, feature_cols=['P', 'C', 'S']) -> str:
    """
    Generate a unique cache key based on target_id and feature matrix hash.
    
    Args:
        target_id: Target company identifier
        features_df: DataFrame with feature columns
        feature_cols: List of feature column names
    
    Returns:
        str: Cache key (filename-safe)
    """
    feature_hash = _hash_features_df(features_df, feature_cols)
    # Create filename-safe key
    safe_target_id = target_id.replace('/', '_').replace('\\', '_').replace(' ', '_')
    return f"{safe_target_id}_{feature_hash}"


def load_cached_shap(target_id: str, features_df, feature_cols=['P', 'C', 'S']) -> Optional[Dict]:
    """
    Load cached SHAP model and values if available and not expired.
    
    Args:
        target_id: Target company identifier
        features_df: DataFrame with feature columns
        feature_cols: List of feature column names
    
    Returns:
        Dict with keys: 'model', 'shap_values', 'base_value', 'metadata', or None if not found/expired
    """
    cache_key = _get_cache_key(target_id, features_df, feature_cols)
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
        
        # Verify feature hash matches (safety check)
        cached_hash = metadata.get('feature_hash')
        current_hash = _hash_features_df(features_df, feature_cols)
        if cached_hash != current_hash:
            # Feature matrix changed - invalidate cache
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            return None
        
        # Load cached model and SHAP values
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        return {
            'model': cached_data.get('model'),
            'shap_values': cached_data.get('shap_values'),
            'base_value': cached_data.get('base_value'),
            'metadata': metadata
        }
    except Exception as e:
        # If loading fails, delete corrupted cache
        cache_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
        return None


def save_cached_shap(
    target_id: str,
    features_df,
    model,
    shap_values: np.ndarray,
    base_value: float,
    feature_cols=['P', 'C', 'S']
):
    """
    Save SHAP model and values to cache.
    
    Args:
        target_id: Target company identifier
        features_df: DataFrame with feature columns
        model: Trained XGBoost model
        shap_values: SHAP values array
        base_value: SHAP base value
        feature_cols: List of feature column names
    """
    cache_key = _get_cache_key(target_id, features_df, feature_cols)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    metadata_file = CACHE_DIR / f"{cache_key}_metadata.json"
    
    try:
        # Save model and SHAP values
        cached_data = {
            'model': model,
            'shap_values': shap_values,
            'base_value': base_value
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        # Save metadata
        feature_hash = _hash_features_df(features_df, feature_cols)
        metadata = {
            'target_id': target_id,
            'feature_hash': feature_hash,
            'feature_cols': feature_cols,
            'n_candidates': len(features_df),
            'created_at': datetime.now().isoformat(),
            'cache_key': cache_key
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        # Non-critical - continue even if caching fails
        pass


def clear_expired_shap_cache() -> int:
    """
    Clear expired SHAP cache files.
    
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

