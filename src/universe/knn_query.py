import os
import numpy as np
import pandas as pd
from time import sleep
import faiss
import hashlib

try:
    import openai
except ImportError:
    openai = None

MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../../data/cache/embedding")
FAISS_PATH = os.path.join(os.path.dirname(__file__), "../../data/embeddings/universe_index.faiss")
META_PATH = os.path.join(os.path.dirname(__file__), "../../data/embeddings/universe_meta.parquet")

def preprocess(text):
    import re
    text = (text or "").lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:8000]

def embedding_cache_path(text, model=MODEL):
    h = hashlib.sha256((text + model).encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.npy")

def get_query_embedding(query, run_with_openai=False, api_key=None):
    txt = preprocess(query)
    path = embedding_cache_path(txt)
    if os.path.isfile(path):
        return np.load(path)
    if run_with_openai:
        if openai is None:
            raise RuntimeError("openai package not installed!")
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        try:
            # Try new OpenAI client API first
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            resp = client.embeddings.create(input=[txt], model=MODEL)
            v = np.array(resp.data[0].embedding, dtype=np.float32)
        except (ImportError, AttributeError):
            # Fallback to old API
            resp = openai.Embedding.create(input=[txt], model=MODEL)
            v = np.array(resp['data'][0]['embedding'], dtype=np.float32)
        v /= np.linalg.norm(v)
        np.save(path, v)
        return v
    else:
        # Use deterministic random for testing (same seed as embeddings_index)
        np.random.seed(hash(txt) % (2**31))  # Deterministic based on text
        v = np.random.randn(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        np.save(path, v)
        return v

def main(query_text, topk=10, run_with_openai=False):
    print("Loading FAISS index and metadata...")
    if not os.path.exists(FAISS_PATH):
        print(f"Error: FAISS index not found at {FAISS_PATH}")
        print("Please run: python comps/src/universe/embeddings_index.py first")
        return
    if not os.path.exists(META_PATH):
        print(f"Error: Metadata file not found at {META_PATH}")
        print("Please run: python comps/src/universe/embeddings_index.py first")
        return
    
    index = faiss.read_index(FAISS_PATH)
    meta = pd.read_parquet(META_PATH)
    print(f"✓ Loaded index with {index.ntotal} vectors and {len(meta)} companies")
    
    print("Embedding the target query...")
    query_vec = get_query_embedding(query_text, run_with_openai)
    query_vec = query_vec.reshape(1, -1)
    
    print("Searching FAISS index...")
    D, I = index.search(query_vec, min(topk, index.ntotal))
    
    print(f"\n{'='*80}")
    print(f"Top-{len(I[0])} semantic matches for:")
    print(f"\"{query_text}\"")
    print(f"{'='*80}\n")
    
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        row = meta.iloc[idx]
        exchange = row.get('exchange', 'N/A')
        sector = row.get('sector', 'N/A')
        industry = row.get('industry', 'N/A')
        summary = row.get('summary', '')[:150] or 'No description available'
        
        print(f"{rank:2d}. {row['ticker']:6s} | {row['name'][:40]:40s} | {exchange:8s}")
        print(f"     Sector: {sector} | Industry: {industry}")
        print(f"     Score: {score:.4f}")
        print(f"     {summary}...")
        print()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Free text to embed/query")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--openai", action='store_true', help="Use OpenAI for real embedding (else uses random)")
    args = parser.parse_args()
    main(args.query, args.topk, args.openai)
