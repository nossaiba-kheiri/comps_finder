"""
model_similarity.py: Compute revenue/delivery model similarity to target (generic, no industry bias).

This module converts revenue and delivery models into vectors and computes similarity to target.
Uses 3-layer model:
- Layer 1: 4 economic archetypes (universal)
- Layer 2: Extensible revenue channels
- Layer 3: Delivery modes

Works for ANY industry (consulting, SaaS, marketplace, hardware, etc.) - no hardcoding.
"""
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Load schema from config
# Calculate path relative to comps/ root directory
# __file__ is at: comps/src/features/model_similarity.py
# We need: comps/config/model_schema.yaml
# So: go up 3 levels (features -> src -> comps), then into config/
_current_file = Path(__file__).resolve()
_comps_root = _current_file.parent.parent.parent  # comps/
SCHEMA_PATH = _comps_root / 'config' / 'model_schema.yaml'

try:
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, 'r') as f:
            SCHEMA = yaml.safe_load(f)
        ARCHETYPES = SCHEMA.get('archetypes', [])
        REVENUE_CHANNELS = SCHEMA.get('channels', [])
        DELIVERY_MODES = SCHEMA.get('delivery_modes', [])
    else:
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")
except Exception as e:
    # Fallback to defaults if schema file not found
    print(f"Warning: Could not load model_schema.yaml from {SCHEMA_PATH}: {e}, using defaults")
    ARCHETYPES = [
        "unit_of_work",
        "access_capability",
        "performance_outcome",
        "intermediation"
    ]
    REVENUE_CHANNELS = [
        "license_upfront",
        "subscription_recurring",
        "usage_based",
        "transaction_fees",
        "professional_services_project",
        "managed_services",
        "data_license",
        "commission_take_rate",
        "marketplace_take_rate",
        "hardware_sales",
        "consumables_replace",
        "financing_fee",
        "advertising",
        "embedded_finance",
        "grants",
        "government_contracts_fixed",
        "government_contracts_time_material",
        "enterprise_custom_deal",
        "tokenomics",
        "other",
    ]
    DELIVERY_MODES = [
        "on_premise_software",
        "cloud_saas",
        "field_services",
        "remote_services",
        "retail_distribution",
        "online_marketplace",
        "embedded_in_partner_product",
        "api_access",
        "data_feed",
        "other",
    ]


def archetype_vector(revenue_archetypes: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Convert revenue_archetypes dict to normalized vector (Layer 1).
    
    Args:
        revenue_archetypes: Dict mapping archetype names to percentages (0.0-1.0)
                           e.g., {"unit_of_work": 0.7, "access_capability": 0.3}
    
    Returns:
        numpy array of length len(ARCHETYPES), normalized to sum to 1.0
    """
    if not revenue_archetypes:
        # Default: equal distribution if missing
        vec = np.ones(len(ARCHETYPES), dtype=float) / len(ARCHETYPES)
        return vec
    
    vec = np.array(
        [float(revenue_archetypes.get(arch, 0.0) or 0.0) for arch in ARCHETYPES],
        dtype=float,
    )
    
    # Normalize to sum to 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    else:
        # If all zeros, default to equal distribution
        vec = np.ones(len(ARCHETYPES), dtype=float) / len(ARCHETYPES)
    
    return vec


def channel_vector(revenue_channels: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Convert revenue_channels dict to normalized vector (Layer 2).
    
    Args:
        revenue_channels: Dict mapping channel names to percentages (0.0-1.0)
                         e.g., {"subscription_recurring": 0.8, "professional_services_project": 0.2}
    
    Returns:
        numpy array of length len(REVENUE_CHANNELS), normalized to sum to 1.0
    """
    if not revenue_channels:
        # Default: all "other" if missing
        vec = np.zeros(len(REVENUE_CHANNELS), dtype=float)
        if "other" in REVENUE_CHANNELS:
            vec[REVENUE_CHANNELS.index("other")] = 1.0
        else:
            vec[-1] = 1.0
        return vec
    
    vec = np.array(
        [float(revenue_channels.get(channel, 0.0) or 0.0) for channel in REVENUE_CHANNELS],
        dtype=float,
    )
    
    # Normalize to sum to 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    else:
        # If all zeros, default to "other"
        if "other" in REVENUE_CHANNELS:
            vec[REVENUE_CHANNELS.index("other")] = 1.0
        else:
            vec[-1] = 1.0
    
    return vec


def revenue_vector(revenue_model_mix: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Maps old revenue_model_mix to new channel_vector format.
    """
    return channel_vector(revenue_model_mix)


def delivery_vector(delivery_modes: Optional[List[str]]) -> np.ndarray:
    """
    Convert delivery_modes list to binary vector.
    
    Args:
        delivery_modes: List of delivery mode strings
                       e.g., ["cloud_saas", "remote_services"]
    
    Returns:
        numpy array of length len(DELIVERY_MODES), binary (1.0 if mode present, 0.0 otherwise)
    """
    if not delivery_modes:
        return np.zeros(len(DELIVERY_MODES), dtype=float)
    
    modes_set = {str(m).lower().strip() for m in delivery_modes if m}
    return np.array(
        [1.0 if bucket.lower() in modes_set else 0.0 for bucket in DELIVERY_MODES],
        dtype=float,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity in [0, 1] (1.0 = identical, 0.0 = orthogonal)
    """
    if len(a) != len(b):
        return 0.0
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    dot_product = np.dot(a, b)
    return float(dot_product / (norm_a * norm_b))


def compute_model_similarities(
    candidate_data: Dict,
    target_profile: Dict
) -> Dict[str, float]:
    """
    Compute revenue and delivery model similarities between candidate and target (3-layer model).
    
    Args:
        candidate_data: Dict with revenue_archetypes, revenue_channels, delivery_modes (from LLM extraction)
        target_profile: Target JSON dict with revenue_archetypes, revenue_channels, delivery_modes
    
    Returns:
        Dict with:
            - archetype_similarity: float [0, 1] (Layer 1)
            - channel_similarity: float [0, 1] (Layer 2)
            - delivery_mode_similarity: float [0, 1] (Layer 3)
            - revenue_model_similarity: float [0, 1] (backward compatibility - uses channel_similarity)
    """
    # Layer 1: Archetypes
    cand_archetypes = candidate_data.get('revenue_archetypes', {})
    target_archetypes = target_profile.get('revenue_archetypes', {})
    
    # Layer 2: Channels
    cand_channels = candidate_data.get('revenue_channels', candidate_data.get('revenue_model_mix', {}))
    target_channels = target_profile.get('revenue_channels', target_profile.get('revenue_model_mix', {}))
    
    # Layer 3: Delivery modes
    cand_delivery_modes = candidate_data.get('delivery_modes', [])
    target_delivery_modes = target_profile.get('delivery_modes', [])
    
    # Convert to vectors
    cand_arch_vec = archetype_vector(cand_archetypes)
    target_arch_vec = archetype_vector(target_archetypes)
    
    cand_chan_vec = channel_vector(cand_channels)
    target_chan_vec = channel_vector(target_channels)
    
    cand_del_vec = delivery_vector(cand_delivery_modes)
    target_del_vec = delivery_vector(target_delivery_modes)
    
    # Compute similarities
    archetype_sim = cosine_similarity(cand_arch_vec, target_arch_vec)
    channel_sim = cosine_similarity(cand_chan_vec, target_chan_vec)
    delivery_sim = cosine_similarity(cand_del_vec, target_del_vec)
    
    return {
        'archetype_similarity': archetype_sim,
        'channel_similarity': channel_sim,
        'delivery_mode_similarity': delivery_sim,
        'revenue_model_similarity': channel_sim,  # Backward compatibility
    }


def convert_legacy_revenue_model(revenue_model: List[str]) -> Dict[str, float]:
    """
    Convert legacy revenue_model list format to revenue_model_mix dict.
    
    This is a backward compatibility function to convert the old format:
    ["project_fees", "time_and_materials", "subscription_software"]
    
    To the new format:
    {"professional_services_project": 0.5, "recurring_subscription": 0.5}
    
    Args:
        revenue_model: List of revenue model strings (legacy format)
    
    Returns:
        Dict mapping revenue buckets to estimated percentages
    """
    if not revenue_model:
        return {"other": 1.0}
    
    revenue_model_lower = [str(rm).lower().strip() for rm in revenue_model if rm]
    
    # Mapping from legacy format to new buckets
    mapping = {
        'project_fees': 'professional_services_project',
        'time_and_materials': 'professional_services_project',
        'retainers': 'managed_services_recurring',
        'managed_services': 'managed_services_recurring',
        'subscription_software': 'recurring_subscription',
        'perpetual_license': 'one_time_license',
        'usage_based': 'usage_based',
        'transaction_fees': 'transaction_fees',
    }
    
    # Count occurrences
    bucket_counts = {}
    for rm in revenue_model_lower:
        # Try direct match first
        if rm in mapping:
            bucket = mapping[rm]
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        else:
            # Try partial match
            matched = False
            for legacy_key, bucket in mapping.items():
                if legacy_key in rm or rm in legacy_key:
                    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                    matched = True
                    break
            if not matched:
                bucket_counts['other'] = bucket_counts.get('other', 0) + 1
    
    # Convert counts to percentages
    total = sum(bucket_counts.values())
    if total == 0:
        return {"other": 1.0}
    
    return {bucket: count / total for bucket, count in bucket_counts.items()}


def infer_archetypes_from_legacy(
    business_model_type: Optional[str],
    revenue_model: List[str],
    services_share: float
) -> Dict[str, float]:
    """
    Infer revenue_archetypes from legacy business_model_type and revenue_model.
    
    This is a backward compatibility function to convert old format to new 3-layer model.
    
    Args:
        business_model_type: Legacy business_model_type string
        revenue_model: Legacy revenue_model list
        services_share: services_share_estimate (0.0-1.0)
    
    Returns:
        Dict mapping archetype names to estimated percentages
    """
    archetypes = {
        "unit_of_work": 0.0,
        "access_capability": 0.0,
        "performance_outcome": 0.0,
        "intermediation": 0.0
    }
    
    revenue_model_lower = [str(rm).lower().strip() for rm in revenue_model if rm]
    business_model_lower = str(business_model_type or '').lower()
    
    # Infer from business_model_type
    if business_model_lower in ['services'] or services_share >= 0.7:
        archetypes["unit_of_work"] = 0.7
        archetypes["access_capability"] = 0.2
        archetypes["performance_outcome"] = 0.1
    elif business_model_lower in ['software'] or services_share <= 0.2:
        archetypes["access_capability"] = 0.8
        archetypes["unit_of_work"] = 0.1
        archetypes["performance_outcome"] = 0.1
    elif business_model_lower == 'marketplace':
        archetypes["intermediation"] = 0.8
        archetypes["access_capability"] = 0.1
        archetypes["unit_of_work"] = 0.1
    elif business_model_lower == 'hybrid_services_software':
        # Hybrid: split based on services_share
        archetypes["unit_of_work"] = services_share
        archetypes["access_capability"] = 1.0 - services_share
    else:
        # Default: balanced
        archetypes["unit_of_work"] = 0.4
        archetypes["access_capability"] = 0.4
        archetypes["performance_outcome"] = 0.1
        archetypes["intermediation"] = 0.1
    
    # Adjust based on revenue_model signals
    if 'transaction_fees' in revenue_model_lower or 'marketplace' in revenue_model_lower:
        archetypes["intermediation"] = max(archetypes["intermediation"], 0.5)
    if 'subscription_software' in revenue_model_lower or 'recurring' in ' '.join(revenue_model_lower):
        archetypes["access_capability"] = max(archetypes["access_capability"], 0.6)
    if 'project_fees' in revenue_model_lower or 'time_and_materials' in revenue_model_lower:
        archetypes["unit_of_work"] = max(archetypes["unit_of_work"], 0.6)
    
    # Normalize to sum to 1.0
    total = sum(archetypes.values())
    if total > 0:
        archetypes = {k: v / total for k, v in archetypes.items()}
    else:
        # Default to equal distribution
        archetypes = {k: 0.25 for k in archetypes.keys()}
    
    return archetypes


def convert_legacy_delivery_modes(
    business_model_type: Optional[str],
    has_software_product: bool,
    has_professional_services: bool,
    has_managed_services: bool
) -> List[str]:
    """
    Infer delivery_modes from legacy business model fields.
    
    This is a backward compatibility function.
    
    Args:
        business_model_type: Legacy business_model_type string
        has_software_product: Boolean flag
        has_professional_services: Boolean flag
        has_managed_services: Boolean flag
    
    Returns:
        List of delivery mode strings
    """
    modes = []
    
    if has_software_product:
        if business_model_type and 'software' in str(business_model_type).lower():
            modes.append('cloud_saas')
        else:
            modes.append('on_premise_software')
    
    if has_professional_services or has_managed_services:
        modes.append('remote_services')
        if has_professional_services:
            modes.append('field_services')
    
    if business_model_type and 'marketplace' in str(business_model_type).lower():
        modes.append('online_marketplace')
    
    if business_model_type and 'hardware' in str(business_model_type).lower():
        modes.append('retail_distribution')
    
    if not modes:
        modes.append('other')
    
    return modes

