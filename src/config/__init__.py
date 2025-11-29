"""
Configuration module for business model classification and scoring.

Exports BusinessModelConfig and helper functions.
"""
from .business_model_config import (
    BusinessModelConfig,
    BM_CONFIG,
    load_business_model_config,
    classify_business_model,
    business_model_similarity_scale,
)

__all__ = [
    'BusinessModelConfig',
    'BM_CONFIG',
    'load_business_model_config',
    'classify_business_model',
    'business_model_similarity_scale',
]

