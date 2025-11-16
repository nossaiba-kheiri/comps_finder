#!/usr/bin/env python3
"""
generate_target_embedding.py: Generate and save target profile embedding to metadata file.

This script:
1. Loads target.json
2. Constructs target_profile_text
3. Generates embedding vector
4. Saves to target_metadata.json (includes vector and metadata)
"""
import json
import os
import sys
import hashlib
import numpy as np
from pathlib import Path

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import embedding functions
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from universe.embeddings_index import preprocess, embedding_cache_path, get_cached_embedding

# Configuration
TARGET_PATH = Path("comps/data/target.json")
METADATA_PATH = Path("comps/data/target_metadata.json")
MODEL = "text-embedding-3-large"
EMBED_DIM = 1536


def construct_target_profile_text(target: dict) -> str:
    """
    Construct target_profile_text from target.json fields.
    
    Priority:
    1. Use raw_profile_text if present (contains website + LinkedIn concatenated text)
    2. Fallback: Construct from structured fields (products, customers, product_mix)
    3. Fallback: Use text_profile if present (legacy format)
    """
    # Priority 1: Use raw_profile_text (contains full website + LinkedIn text)
    if target.get('raw_profile_text'):
        return target['raw_profile_text']
    
    # Priority 2: Construct from structured fields
    parts = []
    
    # Add products/services sentence
    products = target.get('business_activity', []) or target.get('products', [])
    if products:
        if len(products) == 1:
            products_sentence = f"We provide {products[0]}. "
        else:
            products_list = ', '.join(products[:-1]) + f", and {products[-1]}"
            products_sentence = f"We provide {products_list} services. "
        parts.append(products_sentence)
    
    # Add customer segments sentence
    customers = target.get('customer_segment', []) or target.get('customers', [])
    if customers:
        if len(customers) == 1:
            customers_sentence = f"We serve {customers[0]}. "
        else:
            customers_list = ', '.join(customers[:-1]) + f", and {customers[-1]}"
            customers_sentence = f"We serve {customers_list}. "
        parts.append(customers_sentence)
    
    # Add product_mix if present (for structured targets)
    product_mix = target.get('product_mix', {})
    if product_mix:
        items = sorted(product_mix.items(), key=lambda x: x[1], reverse=True)
        mix_parts = []
        for term, weight in items:
            pct = int(weight * 100)
            mix_parts.append(f"{pct}% in {term}")
        if mix_parts:
            parts.append(f"Approximately {', '.join(mix_parts[:-1])}, and {mix_parts[-1]}. ")
    
    # Priority 3: Use text_profile if present (legacy format)
    if target.get('text_profile'):
        parts.append(target['text_profile'])
    
    # Add business description if available
    if target.get('business_description'):
        parts.append(target['business_description'])
    
    target_profile_text = ' '.join(parts)
    return target_profile_text


def generate_target_embedding(target_path=None, metadata_path=None, run_with_openai=False, args=None):
    """
    Generate embedding for target profile and save to metadata file.
    
    Args:
        target_path: Path to target.json (default: comps/data/target.json)
        metadata_path: Path to save metadata (default: comps/data/target_metadata.json)
        run_with_openai: Whether to use OpenAI API (default: False, uses cached/dummy)
    """
    target_path = target_path or TARGET_PATH
    metadata_path = metadata_path or METADATA_PATH
    
    # Load target.json
    print("="*80)
    print("Generating Target Profile Embedding")
    print("="*80)
    print(f"Loading target from: {target_path}")
    
    if not target_path.exists():
        print(f"Error: {target_path} not found!")
        return None
    
    with open(target_path, 'r') as f:
        target = json.load(f)
    
    print(f"Target: {target.get('name', 'Unknown')}")
    print()
    
    # Construct target profile text
    print("Constructing target profile text...")
    target_profile_text = construct_target_profile_text(target)
    print(f"Profile text length: {len(target_profile_text)} characters")
    print(f"Preview: {target_profile_text[:200]}...")
    print()
    
    # Preprocess text
    text_clean = preprocess(target_profile_text)
    
    # Generate embedding
    print("Generating embedding...")
    api_key = os.getenv("OPENAI_API_KEY") if run_with_openai else None
    
    if run_with_openai and not api_key:
        print("Warning: OPENAI_API_KEY not set, using cached/dummy embedding")
        run_with_openai = False
    
    embedding = get_cached_embedding(
        text_clean,
        model=MODEL,
        api_key=api_key,
        run_with_openai=run_with_openai
    )
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print()
    
    # Create metadata structure
    target_id = target.get('target_id', 'unknown')
    target_metadata = target.get('metadata', {})
    
    metadata = {
        "target_id": target_id,  # Unique identifier from target.json
        "target_name": target.get('name', 'Unknown'),
        "target_url": target.get('url', ''),
        "generated_at": None,  # Will be set by datetime
        "embedding_model": MODEL,
        "embedding_dim": EMBED_DIM,
        "target_profile_text": target_profile_text,
        "target_profile_text_clean": text_clean,
        "embedding_vector": embedding.tolist(),  # Convert numpy array to list for JSON
        "target_metadata": {
            "products": target.get('products', []),
            "customers": target.get('customers', []),
            "industry": target.get('industry', ''),
            "mode": target.get('mode', 'all_segments'),
            "product_mix": target.get('product_mix', {}),
            "business_activity": target.get('business_activity', []),
            "customer_segment": target.get('customer_segment', []),
            # Include original target metadata
            "created_at": target_metadata.get('created_at', ''),
            "created_by": target_metadata.get('created_by', ''),
            "extraction_method": target_metadata.get('extraction_method', ''),
            "sic_code": target_metadata.get('sic_code', '')
        }
    }
    
    # Add timestamp
    from datetime import datetime
    metadata["generated_at"] = datetime.now().isoformat()
    
    # Save metadata - use script's directory to avoid path issues
    script_dir = Path(__file__).parent  # comps/data/
    if not args or not hasattr(args, 'output') or not args.output:
        # If no explicit output path, save to script directory
        metadata_path = script_dir / "target_metadata.json"
    
    print(f"Saving metadata to: {metadata_path}")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved target metadata to {metadata_path}")
    print()
    
    # Also save embedding as numpy file for direct loading (use target_id in filename if available)
    # Save to script directory to avoid path issues
    if target_id and target_id != 'unknown':
        embedding_path = script_dir / f"target_{target_id}_embedding.npy"
    else:
        embedding_path = script_dir / f"{metadata_path.stem}_embedding.npy"
    np.save(embedding_path, embedding)
    print(f"✓ Saved embedding vector to {embedding_path}")
    print()
    
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Target ID: {target_id}")
    print(f"Target: {metadata['target_name']}")
    print(f"Profile text: {len(target_profile_text)} chars")
    print(f"Embedding: {embedding.shape} (norm: {np.linalg.norm(embedding):.4f})")
    print(f"Metadata file: {metadata_path}")
    print(f"Embedding file: {embedding_path}")
    print("="*80)
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate and save target profile embedding')
    parser.add_argument('--target', type=str, default=None, help='Path to target.json')
    parser.add_argument('--output', type=str, default=None, help='Path to save metadata.json')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI API (requires OPENAI_API_KEY)')
    
    args = parser.parse_args()
    
    generate_target_embedding(
        target_path=Path(args.target) if args.target else None,
        metadata_path=Path(args.output) if args.output else None,
        run_with_openai=args.openai,
        args=args
    )

