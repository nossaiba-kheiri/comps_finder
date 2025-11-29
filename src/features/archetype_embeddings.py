"""
archetype_embeddings.py: Domain-specific archetype embeddings learned from corpus.

This module implements domain-embedding (not hardcoding) by:
1. Extracting domain-specific vocabulary from target company corpus
2. Clustering terms into latent archetypes
3. Creating archetype vectors for each company
4. Computing similarity on archetype vectors (not generic sector tags)

This distinguishes operating models (e.g., "revenue cycle consulting" vs "healthcare staffing")
without hardcoding industry-specific rules.
"""
import numpy as np
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
import re
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Import embedding function
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
    from universe.embeddings_index import get_cached_embedding
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    get_cached_embedding = None

# Cache configuration
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "archetype_embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRATION_DAYS = 30


def extract_domain_vocabulary(
    corpus_text: str,
    min_term_length: int = 3,
    max_terms: int = 200
) -> List[str]:
    """
    Extract domain-specific vocabulary from corpus text.
    
    Extracts noun phrases, technical terms, and domain-specific concepts
    that represent operating model vocabulary (not generic sector tags).
    
    Args:
        corpus_text: Combined text from website, whitepapers, case studies, etc.
        min_term_length: Minimum length of extracted terms
        max_terms: Maximum number of terms to extract
    
    Returns:
        List of domain-specific terms (e.g., "revenue cycle", "EHR implementation")
    """
    if not corpus_text:
        return []
    
    text_lower = corpus_text.lower()
    
    # Extract noun phrases (2-4 word sequences)
    # Pattern: adjective* noun+ (e.g., "revenue cycle", "enterprise health record")
    noun_phrase_pattern = r'\b(?:[a-z]+(?:\s+[a-z]+){1,3})\b'
    noun_phrases = re.findall(noun_phrase_pattern, text_lower)
    
    # Filter by length and common patterns
    domain_terms = []
    for phrase in noun_phrases:
        words = phrase.split()
        # Keep phrases that:
        # - Are 2-4 words (technical terms are usually multi-word)
        # - Don't start with common stop words
        # - Contain domain-relevant words
        if 2 <= len(words) <= 4:
            # Filter out common stop phrases
            stop_starts = ['the ', 'a ', 'an ', 'and ', 'or ', 'for ', 'with ', 'from ']
            if not any(phrase.startswith(stop) for stop in stop_starts):
                domain_terms.append(phrase.strip())
    
    # Count frequency and take most common
    term_counts = Counter(domain_terms)
    
    # Filter by minimum frequency (appears at least 2 times)
    filtered_terms = [term for term, count in term_counts.most_common(max_terms) if count >= 2]
    
    # Also extract single-word technical terms (longer words, likely domain-specific)
    words = re.findall(r'\b[a-z]{6,}\b', text_lower)  # Words 6+ chars
    word_counts = Counter(words)
    # Filter common English words
    common_words = {'company', 'services', 'solutions', 'technology', 'business', 'management'}
    technical_words = [w for w, count in word_counts.most_common(50) 
                      if w not in common_words and count >= 2]
    
    # Combine and deduplicate
    all_terms = list(set(filtered_terms + technical_words))
    
    return all_terms[:max_terms]


def cluster_archetypes(
    terms: List[str],
    embeddings: Dict[str, np.ndarray],
    n_clusters: int = 5,
    run_with_openai: bool = False
) -> Dict[int, List[str]]:
    """
    Cluster domain terms into latent archetypes using K-means on embeddings.
    
    This learns archetypes from data, not preset rules.
    
    Args:
        terms: List of domain-specific terms
        embeddings: Dict mapping terms to embedding vectors
        n_clusters: Number of archetype clusters to learn
        run_with_openai: Whether to use OpenAI embeddings
    
    Returns:
        Dict mapping cluster_id to list of terms in that archetype
    """
    if not terms or not embeddings:
        return {}
    
    # Get embeddings for all terms
    term_vectors = []
    valid_terms = []
    
    for term in terms:
        if term in embeddings:
            term_vectors.append(embeddings[term])
            valid_terms.append(term)
        elif EMBEDDINGS_AVAILABLE:
            # Try to get embedding if not cached
            emb = get_cached_embedding(term, run_with_openai=run_with_openai)
            if emb is not None:
                embeddings[term] = emb
                term_vectors.append(emb)
                valid_terms.append(term)
    
    if len(term_vectors) < n_clusters:
        # Not enough terms to cluster - return single cluster
        return {0: valid_terms}
    
    # Cluster using K-means
    X = np.array(term_vectors)
    kmeans = KMeans(n_clusters=min(n_clusters, len(term_vectors)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Group terms by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(valid_terms[idx])
    
    return clusters


def build_archetype_vectors(
    company_corpus: str,
    company_id: str = None,
    n_archetypes: int = 5,
    run_with_openai: bool = False,
    use_cache: bool = True
) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    """
    Build archetype vectors for a company from its corpus.
    
    This is the core function that learns domain-specific archetypes.
    Checks cache first, then computes if needed.
    
    Args:
        company_corpus: Combined text from company (website, docs, etc.)
        company_id: Company identifier (ticker, name, or target_id) for caching
        n_archetypes: Number of archetype clusters to learn
        run_with_openai: Whether to use OpenAI embeddings
        use_cache: Whether to use cache (default: True)
    
    Returns:
        Tuple of:
        - archetype_vectors: np.ndarray of shape (n_archetypes, embedding_dim)
        - archetype_terms: Dict mapping archetype_id to list of terms
    """
    if not company_corpus:
        # Return zero vectors if no corpus
        if EMBEDDINGS_AVAILABLE:
            # Get embedding dimension from a dummy embedding
            dummy_emb = get_cached_embedding("test", run_with_openai=run_with_openai)
            if dummy_emb is not None:
                dim = len(dummy_emb)
                return np.zeros((n_archetypes, dim)), {}
        return np.array([]), {}
    
    # Check cache first (if enabled and company_id provided)
    if use_cache and company_id:
        cached_result = load_cached_archetypes(company_id, company_corpus, n_archetypes)
        if cached_result is not None:
            archetype_vectors, archetype_terms = cached_result
            if archetype_vectors.size > 0:
                return archetype_vectors, archetype_terms
    
    # Step 1: Extract domain vocabulary
    domain_terms = extract_domain_vocabulary(company_corpus)
    
    if not domain_terms:
        return np.array([]), {}
    
    # Step 2: Get embeddings for all terms
    term_embeddings = {}
    for term in domain_terms:
        if EMBEDDINGS_AVAILABLE:
            emb = get_cached_embedding(term, run_with_openai=run_with_openai)
            if emb is not None:
                term_embeddings[term] = emb
    
    if not term_embeddings:
        return np.array([]), {}
    
    # Step 3: Cluster terms into archetypes
    archetype_terms = cluster_archetypes(
        domain_terms,
        term_embeddings,
        n_clusters=n_archetypes,
        run_with_openai=run_with_openai
    )
    
    # Step 4: Create archetype vectors (mean of term embeddings in each cluster)
    archetype_vectors = []
    for cluster_id in sorted(archetype_terms.keys()):
        cluster_terms = archetype_terms[cluster_id]
        cluster_embs = [term_embeddings[term] for term in cluster_terms if term in term_embeddings]
        if cluster_embs:
            archetype_vec = np.mean(cluster_embs, axis=0)
            archetype_vectors.append(archetype_vec)
        else:
            # Empty cluster - use zero vector
            if archetype_vectors:
                dim = len(archetype_vectors[0])
                archetype_vectors.append(np.zeros(dim))
            else:
                # First cluster is empty - get dim from first embedding
                if term_embeddings:
                    dim = len(next(iter(term_embeddings.values())))
                    archetype_vectors.append(np.zeros(dim))
    
    if not archetype_vectors:
        return np.array([]), {}
    
    archetype_vectors_array = np.array(archetype_vectors)
    
    # Save to cache (if enabled and company_id provided)
    if use_cache and company_id and archetype_vectors_array.size > 0:
        save_cached_archetypes(
            company_id,
            company_corpus,
            archetype_vectors_array,
            archetype_terms,
            n_archetypes
        )
    
    return archetype_vectors_array, archetype_terms


def compute_archetype_similarity(
    target_archetypes: np.ndarray,
    candidate_archetypes: np.ndarray
) -> float:
    """
    Compute similarity between target and candidate archetype vectors.
    
    Uses optimal matching (Hungarian algorithm approximation) or mean cosine similarity.
    
    Args:
        target_archetypes: np.ndarray of shape (n_archetypes, embedding_dim)
        candidate_archetypes: np.ndarray of shape (n_archetypes, embedding_dim)
    
    Returns:
        float [0, 1]: Similarity score
    """
    if target_archetypes.size == 0 or candidate_archetypes.size == 0:
        return 0.0
    
    # Normalize archetype vectors
    def normalize_vec(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    
    target_norm = np.array([normalize_vec(v) for v in target_archetypes])
    candidate_norm = np.array([normalize_vec(v) for v in candidate_archetypes])
    
    # Compute pairwise cosine similarities
    # Shape: (n_target_archetypes, n_candidate_archetypes)
    similarities = cosine_similarity(target_norm, candidate_norm)
    
    # Use max similarity for each target archetype (best match)
    # Then average across target archetypes
    max_similarities = np.max(similarities, axis=1)
    mean_similarity = np.mean(max_similarities)
    
    # Map from [-1, 1] to [0, 1]
    return float((mean_similarity + 1.0) / 2.0)


def get_company_corpus(company_data: Dict) -> str:
    """
    Extract corpus text from company data (for archetype learning).
    
    Combines text from:
    - Website content
    - Business description
    - Products/services descriptions
    - Evidence quotes
    
    Args:
        company_data: Company dict with various text fields
    
    Returns:
        Combined corpus text
    """
    corpus_parts = []
    
    # Business description
    if company_data.get('business_description'):
        corpus_parts.append(company_data['business_description'])
    
    # Business activity
    if company_data.get('business_activity'):
        if isinstance(company_data['business_activity'], list):
            corpus_parts.extend(company_data['business_activity'])
        else:
            corpus_parts.append(str(company_data['business_activity']))
    
    # Products
    if company_data.get('products'):
        if isinstance(company_data['products'], list):
            corpus_parts.extend(company_data['products'])
        else:
            corpus_parts.append(str(company_data['products']))
    
    # Evidence quotes
    if company_data.get('evidence'):
        if isinstance(company_data['evidence'], list):
            for ev in company_data['evidence']:
                if isinstance(ev, dict) and ev.get('quote'):
                    corpus_parts.append(ev['quote'])
                elif isinstance(ev, str):
                    corpus_parts.append(ev)
        elif isinstance(company_data['evidence'], str):
            corpus_parts.append(company_data['evidence'])
    
    # Raw profile text (if available)
    if company_data.get('raw_profile_text'):
        corpus_parts.append(company_data['raw_profile_text'])
    
    return ' '.join(corpus_parts)


def _hash_corpus(corpus_text: str) -> str:
    """
    Generate a hash of the corpus text for cache key.
    
    Args:
        corpus_text: Corpus text string
    
    Returns:
        str: Hex digest of the hash
    """
    hash_obj = hashlib.sha256(corpus_text.encode('utf-8'))
    return hash_obj.hexdigest()


def _get_cache_key(company_id: str, corpus_text: str, n_archetypes: int = 5) -> str:
    """
    Generate a unique cache key based on company ID, corpus hash, and n_archetypes.
    
    Args:
        company_id: Company identifier (ticker, name, or target_id)
        corpus_text: Corpus text (will be hashed)
        n_archetypes: Number of archetypes
    
    Returns:
        str: Cache key (filename-safe)
    """
    corpus_hash = _hash_corpus(corpus_text)
    safe_id = company_id.replace('/', '_').replace('\\', '_').replace(' ', '_')
    return f"{safe_id}_{corpus_hash[:16]}_{n_archetypes}"


def load_cached_archetypes(
    company_id: str,
    corpus_text: str,
    n_archetypes: int = 5
) -> Optional[Tuple[np.ndarray, Dict[int, List[str]]]]:
    """
    Load cached archetype vectors if available and not expired.
    
    Args:
        company_id: Company identifier (ticker, name, or target_id)
        corpus_text: Corpus text (for hash verification)
        n_archetypes: Number of archetypes
    
    Returns:
        Tuple of (archetype_vectors, archetype_terms) or None if not found/expired
    """
    cache_key = _get_cache_key(company_id, corpus_text, n_archetypes)
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
        
        # Verify corpus hash matches (safety check)
        cached_hash = metadata.get('corpus_hash')
        current_hash = _hash_corpus(corpus_text)
        if cached_hash != current_hash:
            # Corpus changed - invalidate cache
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            return None
        
        # Load cached archetype vectors and terms
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        return (
            cached_data.get('archetype_vectors'),
            cached_data.get('archetype_terms', {})
        )
    except Exception as e:
        # If loading fails, delete corrupted cache
        cache_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
        return None


def save_cached_archetypes(
    company_id: str,
    corpus_text: str,
    archetype_vectors: np.ndarray,
    archetype_terms: Dict[int, List[str]],
    n_archetypes: int = 5
):
    """
    Save archetype vectors to cache.
    
    Args:
        company_id: Company identifier (ticker, name, or target_id)
        corpus_text: Corpus text (for hash verification)
        archetype_vectors: np.ndarray of archetype vectors
        archetype_terms: Dict mapping archetype_id to list of terms
        n_archetypes: Number of archetypes
    """
    cache_key = _get_cache_key(company_id, corpus_text, n_archetypes)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    metadata_file = CACHE_DIR / f"{cache_key}_metadata.json"
    
    try:
        # Save archetype vectors and terms
        cached_data = {
            'archetype_vectors': archetype_vectors,
            'archetype_terms': archetype_terms
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        # Save metadata
        corpus_hash = _hash_corpus(corpus_text)
        metadata = {
            'company_id': company_id,
            'corpus_hash': corpus_hash,
            'n_archetypes': n_archetypes,
            'created_at': datetime.now().isoformat(),
            'cache_key': cache_key,
            'archetype_count': len(archetype_terms)
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        # Non-critical - continue even if caching fails
        pass


def clear_expired_archetype_cache() -> int:
    """
    Clear expired archetype cache files.
    
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

