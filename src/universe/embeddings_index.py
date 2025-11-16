"""
embeddings_index.py: Build/query FAISS or HNSW embedding indexes and return top-K neighbors.
"""

import os
import sys
import hashlib
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
import faiss

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

try:
    import openai
except ImportError:
    openai = None  # User must pip install openai

MODEL = 'text-embedding-3-large'
EMBED_DIM = 3072
BATCH_SIZE = 100
CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../data/cache/embedding')
UNIVERSE_PATH = os.path.join(os.path.dirname(__file__), '../../data/universe_us.csv')
FAISS_PATH = os.path.join(os.path.dirname(__file__), '../../data/embeddings/universe_index.faiss')
META_PATH = os.path.join(os.path.dirname(__file__), '../../data/embeddings/universe_meta.parquet')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)


def preprocess(text):
    import re
    text = (text or "").lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:8000]  # truncate

def embed_openai_batch(texts, model=MODEL, api_key=None):
    if openai is None:
        raise RuntimeError("openai package not installed!")
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not provided!")
    
    results = []
    try:
        # Try new OpenAI client API first
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            resp = client.embeddings.create(input=batch, model=model)
            for item in resp.data:
                v = np.array(item.embedding, dtype=np.float32)
                v /= np.linalg.norm(v)
                results.append(v)
            sleep(0.7)  # gentle on OpenAI
    except (ImportError, AttributeError):
        # Fallback to old API
        openai.api_key = api_key
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            resp = openai.Embedding.create(input=batch, model=model)
            for d in resp['data']:
                v = np.array(d['embedding'], dtype=np.float32)
                v /= np.linalg.norm(v)
                results.append(v)
            sleep(0.7)  # gentle on OpenAI
    return results

def embedding_cache_path(text, model=MODEL):
    h = hashlib.sha256((text + model).encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.npy")

def get_cached_embedding(text, model=MODEL, api_key=None, run_with_openai=False):
    path = embedding_cache_path(text, model)
    if os.path.isfile(path):
        return np.load(path)
    
    # Generate embedding
    if run_with_openai:
        # Use OpenAI API
        if openai is None:
            raise RuntimeError("openai package not installed!")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key not provided!")
        
        try:
            # Try new OpenAI client API first
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(input=[text], model=model)
            v = np.array(resp.data[0].embedding, dtype=np.float32)
        except (ImportError, AttributeError):
            # Fallback to old API
            openai.api_key = api_key
            resp = openai.Embedding.create(input=[text], model=model)
            v = np.array(resp['data'][0]['embedding'], dtype=np.float32)
        v /= np.linalg.norm(v)
        np.save(path, v)
        return v
    else:
        # Use deterministic random for testing
        np.random.seed(hash(text) % (2**31))
        v = np.random.randn(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        np.save(path, v)
        return v

def build_universe_embeddings(run_with_openai=False, api_key=None, limit=None):
    df = pd.read_csv(UNIVERSE_PATH)
    if limit:
        df = df.head(limit)
    print(f"Processing {len(df)} companies...")
    
    # Preprocess all summaries
    summaries = [preprocess(x) for x in df['summary'].fillna('')]
    
    # Check cache for all, collect what needs embedding
    vectors_by_idx = [None] * len(df)
    to_embed_data = []  # List of (index, text) tuples
    
    for i, txt in enumerate(summaries):
        cache_f = embedding_cache_path(txt)
        if os.path.isfile(cache_f) and txt:  # Only use cache if text exists
            try:
                vectors_by_idx[i] = np.load(cache_f)
            except Exception:
                to_embed_data.append((i, txt))
        else:
            if txt:  # Only embed if text exists
                to_embed_data.append((i, txt))
    
    # Fetch missing embeddings
    if to_embed_data:
        missing_texts = [txt for _, txt in to_embed_data]
        missing_indices = [idx for idx, _ in to_embed_data]
        print(f"Fetching {len(missing_texts)} missing embeddings...")
        
        if not run_with_openai:
            # For testing, use random vectors with deterministic seed for reproducibility
            np.random.seed(42)
            for idx, txt in to_embed_data:
                v = np.random.randn(EMBED_DIM).astype(np.float32)
                v /= np.linalg.norm(v)
                vectors_by_idx[idx] = v
                np.save(embedding_cache_path(txt), v)
        else:
            # Use OpenAI API
            results = embed_openai_batch(missing_texts, api_key=api_key)
            for (idx, txt), v in zip(to_embed_data, results):
                vectors_by_idx[idx] = v
                np.save(embedding_cache_path(txt), v)
    
    # Filter out None vectors (companies without summaries)
    valid_indices = [i for i, v in enumerate(vectors_by_idx) if v is not None]
    valid_vectors = [vectors_by_idx[i] for i in valid_indices]
    
    if not valid_vectors:
        print("Error: No valid embeddings found!")
        return
    
    # Stack vectors and build FAISS index
    vectors_np = np.vstack(valid_vectors)
    print(f"Building FAISS index with {len(valid_vectors)} vectors...")
    
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors_np)
    faiss.write_index(index, FAISS_PATH)
    
    # Create metadata DataFrame with only valid entries
    meta_rows = []
    for i in valid_indices:
        row_dict = df.iloc[i].to_dict()
        row_dict['faiss_idx'] = len(meta_rows)  # Index in FAISS
        meta_rows.append(row_dict)
    
    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(META_PATH, index=False)
    print(f"✓ Built FAISS index: {FAISS_PATH}")
    print(f"✓ Saved metadata: {META_PATH} ({len(meta)} companies)")

# Example CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai', action='store_true', help='Use real OpenAI embeddings (set OPENAI_API_KEY env var)')
    parser.add_argument('--limit', type=int, default=None, help='Dev: limit number of companies.')
    args = parser.parse_args()
    build_universe_embeddings(run_with_openai=args.openai, limit=args.limit)
