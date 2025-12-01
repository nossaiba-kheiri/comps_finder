"""
economic_signature.py: Extract universal economic signature from company evidence.

This module extracts economic structure (HOW companies make money) rather than
customer segments (WHO they sell to). This works universally across all industries:
- Consulting firms (project fees, retainers)
- SaaS companies (subscription, usage)
- Industrial equipment (capital equipment, aftermarket, consumables)
- Marketplaces (transaction fees, commissions)

The economic signature is a vector that captures:
1. Revenue structure (% capital equipment, % aftermarket, % consumables, % software, etc.)
2. IP intensity (proprietary technology, patents, R&D intensity)
3. Customer lock-in (installed base dependency, switching costs, customization)
4. Replacement cycle (capital cycles, consumable cycles)
5. Gross margin tier
6. Asset intensity
"""

from typing import Dict, Optional, List, Tuple
import numpy as np


def extract_economic_signature_from_llm(extracted_data: Dict) -> Dict:
    """
    Extract economic signature from LLM-extracted data.
    
    PRIORITY 1: Use LLM-extracted economic_signature if available
    PRIORITY 2: Infer from revenue_channels and business_model_type
    
    The LLM should extract:
    - capital_equipment_share: % revenue from capital equipment sales
    - aftermarket_service_share: % revenue from aftermarket/service/maintenance
    - consumables_share: % revenue from consumables/parts
    - software_recurring_share: % revenue from software subscriptions/licenses
    - project_services_share: % revenue from project-based services
    - ip_intensity: 0-1 score for IP/proprietary technology intensity
    - customer_lock_in: 0-1 score for switching costs/installed base dependency
    - replacement_cycle_years: Average replacement cycle in years
    - gross_margin_tier: "low" (0-30%), "medium" (30-50%), "high" (50-70%), "very_high" (70%+)
    - asset_intensity: 0-1 score for asset-heavy business model
    
    Args:
        extracted_data: Dict from LLM extraction with revenue_channels, revenue_archetypes, etc.
    
    Returns:
        Dict with economic signature vector components
    """
    # PRIORITY 1: Use LLM-extracted economic_signature if available
    llm_signature = extracted_data.get('economic_signature')
    if llm_signature and isinstance(llm_signature, dict):
        # Validate that it has at least one required field
        if 'capital_equipment_share' in llm_signature or 'ip_intensity' in llm_signature:
            # Merge with defaults for missing fields (best of both worlds)
            inferred = _infer_economic_signature(extracted_data)
            # LLM values take priority, fallback to inferred for missing
            merged = {**inferred, **llm_signature}
            return merged
    
    # PRIORITY 2: Infer from revenue_channels and business_model_type
    return _infer_economic_signature(extracted_data)


def _infer_economic_signature(extracted_data: Dict) -> Dict:
    """
    Infer economic signature from revenue_channels and business_model_type.
    Used as fallback when LLM doesn't provide economic_signature directly.
    """
    # Extract revenue mix from revenue_channels if available
    revenue_channels = extracted_data.get('revenue_channels', {}) or {}
    
    # Map revenue_channels to economic signature components
    capital_equipment_share = revenue_channels.get('hardware_sales', 0.0) or 0.0
    
    # Aftermarket/service = managed_services + professional_services (if equipment-related)
    aftermarket_service_share = (
        (revenue_channels.get('managed_services', 0.0) or 0.0) +
        (revenue_channels.get('professional_services_project', 0.0) or 0.0) * 0.5  # Split with project services
    )
    
    consumables_share = revenue_channels.get('consumables_replace', 0.0) or 0.0
    
    # Software recurring = subscription + licenses
    software_recurring_share = (
        (revenue_channels.get('subscription_recurring', 0.0) or 0.0) +
        (revenue_channels.get('license_upfront', 0.0) or 0.0) * 0.3  # One-time licenses amortized
    )
    
    # IP intensity: Infer from business_model_type, proprietary mentions, R&D focus
    business_model_type = (extracted_data.get('business_model_type') or 'other').lower()
    
    # High IP companies: software, hardware with proprietary tech, hybrid with strong tech
    ip_intensity = 0.5  # Default
    if business_model_type in ['software', 'hardware']:
        ip_intensity = 0.8
    elif business_model_type == 'hybrid_services_software':
        ip_intensity = 0.6
    elif business_model_type == 'services':
        ip_intensity = 0.3  # Lower IP, more human capital
    
    # Customer lock-in: Infer from installed base mentions, customization, switching costs
    # Default based on business model
    customer_lock_in = 0.5
    if capital_equipment_share > 0.5:
        customer_lock_in = 0.8  # Capital equipment has high lock-in
    elif software_recurring_share > 0.5:
        customer_lock_in = 0.7  # Software subscriptions have medium-high lock-in
    elif aftermarket_service_share > 0.3:
        customer_lock_in = 0.6  # Aftermarket dependency indicates lock-in
    
    # Replacement cycle: Default based on business model
    replacement_cycle_years = 3.0  # Default
    if capital_equipment_share > 0.5:
        replacement_cycle_years = 10.0  # Capital equipment: 8-15 years
    elif consumables_share > 0.3:
        replacement_cycle_years = 0.5  # Consumables: monthly/quarterly
    elif software_recurring_share > 0.5:
        replacement_cycle_years = 1.0  # Software: annual subscription
    
    # Gross margin tier: Default based on business model
    gross_margin_tier = "medium"  # Default
    if business_model_type == 'software':
        gross_margin_tier = "high"  # 70-85%
    elif capital_equipment_share > 0.5:
        gross_margin_tier = "medium"  # 30-50%
    elif business_model_type == 'services':
        gross_margin_tier = "medium"  # 30-50%
    
    # Asset intensity: High for capital equipment, low for software/services
    asset_intensity = 0.5
    if capital_equipment_share > 0.5:
        asset_intensity = 0.8  # High asset intensity
    elif software_recurring_share > 0.5:
        asset_intensity = 0.2  # Low asset intensity (software)
    elif business_model_type == 'services':
        asset_intensity = 0.3  # Low-medium (people, not assets)
    
    # Project services share (separate from aftermarket)
    project_services_share = revenue_channels.get('professional_services_project', 0.0) or 0.0
    
    return {
        'capital_equipment_share': float(capital_equipment_share),
        'aftermarket_service_share': float(aftermarket_service_share),
        'consumables_share': float(consumables_share),
        'software_recurring_share': float(software_recurring_share),
        'project_services_share': float(project_services_share),
        'ip_intensity': float(ip_intensity),
        'customer_lock_in': float(customer_lock_in),
        'replacement_cycle_years': float(replacement_cycle_years),
        'gross_margin_tier': gross_margin_tier,
        'asset_intensity': float(asset_intensity)
    }


def economic_signature_to_vector(signature: Dict) -> np.ndarray:
    """
    Convert economic signature dict to normalized vector for similarity comparison.
    
    Vector components (normalized to [0, 1] where applicable):
    0. capital_equipment_share: [0, 1]
    1. aftermarket_service_share: [0, 1]
    2. consumables_share: [0, 1]
    3. software_recurring_share: [0, 1]
    4. ip_intensity: [0, 1]
    5. customer_lock_in: [0, 1]
    6. replacement_cycle_normalized: [0, 1] (normalize 0-20 years → 0-1)
    7. gross_margin_tier_encoded: [0, 1] (low=0.2, medium=0.5, high=0.8, very_high=1.0)
    8. asset_intensity: [0, 1]
    
    Args:
        signature: Dict from extract_economic_signature_from_llm
    
    Returns:
        numpy array of shape (9,) with normalized values
    """
    # Extract components
    capital_equipment = signature.get('capital_equipment_share', 0.0)
    aftermarket = signature.get('aftermarket_service_share', 0.0)
    consumables = signature.get('consumables_share', 0.0)
    software_recurring = signature.get('software_recurring_share', 0.0)
    ip_intensity = signature.get('ip_intensity', 0.5)
    lock_in = signature.get('customer_lock_in', 0.5)
    replacement_cycle = signature.get('replacement_cycle_years', 3.0)
    margin_tier = signature.get('gross_margin_tier', 'medium')
    asset_intensity = signature.get('asset_intensity', 0.5)
    
    # Normalize replacement cycle (0-20 years → 0-1)
    replacement_cycle_normalized = min(replacement_cycle / 20.0, 1.0)
    
    # Encode gross margin tier
    margin_encoding = {
        'low': 0.2,
        'medium': 0.5,
        'high': 0.8,
        'very_high': 1.0
    }
    margin_encoded = margin_encoding.get(margin_tier.lower(), 0.5)
    
    # Build vector
    vector = np.array([
        float(capital_equipment),
        float(aftermarket),
        float(consumables),
        float(software_recurring),
        float(ip_intensity),
        float(lock_in),
        float(replacement_cycle_normalized),
        float(margin_encoded),
        float(asset_intensity)
    ], dtype=np.float32)
    
    return vector


def economic_signature_similarity(
    target_signature: Dict,
    candidate_signature: Dict
) -> float:
    """
    Compute cosine similarity between two economic signature vectors.
    
    This measures HOW similar two companies are in terms of economic structure,
    regardless of customer industries served.
    
    Args:
        target_signature: Economic signature dict for target company
        candidate_signature: Economic signature dict for candidate company
    
    Returns:
        float: Cosine similarity score [0, 1] where 1.0 = identical economic structure
    """
    target_vec = economic_signature_to_vector(target_signature)
    candidate_vec = economic_signature_to_vector(candidate_signature)
    
    # Compute cosine similarity
    dot_product = np.dot(target_vec, candidate_vec)
    norm_target = np.linalg.norm(target_vec)
    norm_candidate = np.linalg.norm(candidate_vec)
    
    if norm_target == 0.0 or norm_candidate == 0.0:
        return 0.0
    
    similarity = dot_product / (norm_target * norm_candidate)
    
    # Ensure in [0, 1] range
    return float(np.clip(similarity, 0.0, 1.0))


def extract_enhanced_revenue_breakdown(extracted_data: Dict) -> Dict:
    """
    Enhanced revenue breakdown extraction with more granular categories for industrial companies.
    
    This function tries to infer more detailed revenue structure from LLM-extracted data
    and business description, especially for capital equipment companies.
    
    Returns additional fields:
    - capital_equipment_share
    - aftermarket_parts_share
    - service_contracts_share
    - consumables_share
    - software_subscription_share
    - project_services_share
    """
    revenue_channels = extracted_data.get('revenue_channels', {}) or {}
    business_model_type = (extracted_data.get('business_model_type') or 'other').lower()
    
    # For hardware/capital equipment companies, try to break down revenue_channels
    # into more specific categories
    
    capital_equipment_share = revenue_channels.get('hardware_sales', 0.0) or 0.0
    
    # Aftermarket parts (if consumables_replace is high, might be parts)
    aftermarket_parts_share = revenue_channels.get('consumables_replace', 0.0) or 0.0
    
    # Service contracts = managed_services (if equipment-related)
    service_contracts_share = revenue_channels.get('managed_services', 0.0) or 0.0
    
    # Consumables (separate from parts if mentioned)
    consumables_share = 0.0  # Will be inferred if aftermarket_parts is high
    
    # Software subscriptions
    software_subscription_share = revenue_channels.get('subscription_recurring', 0.0) or 0.0
    
    # Project services (implementation, installation, custom engineering)
    project_services_share = revenue_channels.get('professional_services_project', 0.0) or 0.0
    
    # If business_model_type is hardware and capital_equipment_share is low,
    # might be missing revenue breakdown - use defaults based on business model
    if business_model_type == 'hardware' and capital_equipment_share < 0.3:
        # Likely capital equipment company but revenue breakdown not extracted properly
        # Use defaults for industrial equipment
        capital_equipment_share = 0.65
        aftermarket_parts_share = 0.15
        service_contracts_share = 0.15
        software_subscription_share = 0.05
    
    return {
        'capital_equipment_share': float(capital_equipment_share),
        'aftermarket_parts_share': float(aftermarket_parts_share),
        'service_contracts_share': float(service_contracts_share),
        'consumables_share': float(consumables_share),
        'software_subscription_share': float(software_subscription_share),
        'project_services_share': float(project_services_share)
    }

