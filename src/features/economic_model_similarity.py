"""
economic_model_similarity.py: Compute economic model similarity using universal economic signature vectors.

This replaces customer-segment-based matching with economic structure matching.
Works universally across all industries by comparing HOW companies make money,
not WHO they sell to.
"""

from typing import Dict, Optional
import numpy as np

try:
    from .economic_signature import (
        extract_economic_signature_from_llm,
        economic_signature_to_vector,
        economic_signature_similarity
    )
except ImportError:
    # Fallback for direct imports
    from features.economic_signature import (
        extract_economic_signature_from_llm,
        economic_signature_to_vector,
        economic_signature_similarity
    )


def compute_economic_model_similarity(
    target_profile: Dict,
    candidate_data: Dict
) -> float:
    """
    Compute economic model similarity between target and candidate using economic signature vectors.
    
    This is a universal similarity metric that works for:
    - Consulting firms (compare project fees vs retainers)
    - SaaS companies (compare subscription vs usage)
    - Industrial equipment (compare capital equipment + aftermarket mix)
    - Marketplaces (compare transaction fee models)
    
    Args:
        target_profile: Target JSON dict with extracted_data or economic_signature
        candidate_data: Candidate extracted_data dict from LLM extraction
    
    Returns:
        float: Economic model similarity score [0, 1] where 1.0 = identical economic structure
    """
    # Extract economic signatures
    # Target might have it in extracted_data or we need to extract it
    if 'economic_signature' in target_profile:
        target_signature = target_profile['economic_signature']
    else:
        # Extract from target's extracted_data or infer from business model
        target_extracted = target_profile.get('extracted_data', target_profile)
        target_signature = extract_economic_signature_from_llm(target_extracted)
    
    # Candidate signature from LLM extraction
    candidate_signature = None
    if 'economic_signature' in candidate_data:
        candidate_signature = candidate_data['economic_signature']
    else:
        # Extract from candidate's extracted_data
        candidate_signature = extract_economic_signature_from_llm(candidate_data)
    
    # Compute similarity
    similarity = economic_signature_similarity(target_signature, candidate_signature)
    
    return float(similarity)


def compute_economic_model_similarity_from_extracted(
    target_extracted: Dict,
    candidate_extracted: Dict
) -> float:
    """
    Compute economic model similarity from extracted data (simpler interface).
    
    Args:
        target_extracted: Target extracted_data dict from LLM
        candidate_extracted: Candidate extracted_data dict from LLM
    
    Returns:
        float: Economic model similarity score [0, 1]
    """
    target_signature = extract_economic_signature_from_llm(target_extracted)
    candidate_signature = extract_economic_signature_from_llm(candidate_extracted)
    
    return float(economic_signature_similarity(target_signature, candidate_signature))

