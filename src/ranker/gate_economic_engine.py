"""
gate_economic_engine.py: Gate to ensure candidates have the same economic engine as target.

The "economic engine" is the core revenue mechanism:
- Awaze: Revenue per night used by travelers (nights/ADR/occupancy)
- REITs: Revenue per square foot leased (square_feet/rent/occupancy)
- Consulting: Revenue per hour worked (hours/time_and_materials/hours_utilized)
- Industrial: Revenue per unit sold (units_sold/product_sale/throughput)

This gate ensures candidates match the target's economic engine family.
"""
from typing import Dict, Any, Optional, Tuple


def infer_economic_engine_from_signature(economic_signature: Dict[str, Any]) -> Optional[str]:
    """
    Infer the economic engine from an economic signature.
    
    Args:
        economic_signature: Dict with capacity_unit, pricing_basis, utilization_metric
    
    Returns:
        str: Economic engine name, or None if cannot determine
    """
    if not economic_signature:
        return None
    
    capacity_unit = str(economic_signature.get('capacity_unit', 'none')).lower()
    pricing_basis = economic_signature.get('pricing_basis', [])
    if isinstance(pricing_basis, str):
        pricing_basis = [pricing_basis]
    pricing_basis_lower = [str(pb).lower() for pb in pricing_basis]
    utilization_metric = str(economic_signature.get('utilization_metric', 'none')).lower()
    
    # Revenue per night used by travelers (hospitality/vacation rentals)
    if capacity_unit == 'nights' and ('adr' in pricing_basis_lower or 'commission' in pricing_basis_lower):
        return 'revenue_per_night'
    
    # Revenue per square foot leased (REITs, real estate)
    if capacity_unit == 'square_feet' and 'rent' in pricing_basis_lower:
        return 'revenue_per_square_foot'
    
    # Revenue per hour worked (consulting, professional services)
    if capacity_unit == 'hours' and ('time_and_materials' in pricing_basis_lower or 'fixed_fee' in pricing_basis_lower):
        return 'revenue_per_hour'
    
    # Revenue per unit sold (industrial, capital equipment)
    if capacity_unit == 'units_sold' and 'product_sale' in pricing_basis_lower:
        return 'revenue_per_unit'
    
    # Revenue per transaction (marketplaces, platforms)
    if 'commission' in pricing_basis_lower and 'transaction' in str(pricing_basis_lower):
        return 'revenue_per_transaction'
    
    # Revenue per subscription (SaaS, software)
    if 'subscription' in pricing_basis_lower or capacity_unit == 'none' and utilization_metric == 'none':
        return 'revenue_per_subscription'
    
    return None


def gate_economic_engine(
    row_or_dict: Dict,
    target_profile: Dict,
    config: Optional[Dict] = None
) -> bool:
    """
    Gate to ensure candidate has the same economic engine as target.
    
    The economic engine is the core revenue mechanism:
    - Awaze: Revenue per night used by travelers (nights/ADR/occupancy)
    - REITs: Revenue per square foot leased (square_feet/rent/occupancy)
    - Consulting: Revenue per hour worked (hours/time_and_materials/hours_utilized)
    - Industrial: Revenue per unit sold (units_sold/product_sale/throughput)
    
    Args:
        row_or_dict: Dict/Series with economic_signature or extracted_data
        target_profile: Target JSON dict with economic_signature
        config: Optional scoring config dict
    
    Returns:
        bool: True if passes gate (same economic engine or cannot determine), False otherwise
    """
    # Get target's economic signature
    target_sig = target_profile.get('extracted_data', {}).get('economic_signature', {}) or target_profile.get('economic_signature', {})
    
    # If target doesn't have economic signature, try to infer from business description
    if not target_sig or not target_sig.get('capacity_unit') or target_sig.get('capacity_unit') == 'none':
        try:
            from features.archetype_inference import infer_archetype_fields_from_target
            inferred = infer_archetype_fields_from_target(target_profile)
            target_sig = {**target_sig, **inferred}
        except Exception:
            # If inference fails, gate not applicable - always pass
            return True
    
    # Infer target's economic engine
    target_engine = infer_economic_engine_from_signature(target_sig)
    if not target_engine:
        # Cannot determine target's economic engine - gate not applicable
        return True
    
    # Get candidate's economic signature
    candidate_sig = {}
    if hasattr(row_or_dict, 'get'):
        # Try extracted_data first
        extracted = row_or_dict.get('extracted_data', {}) or {}
        candidate_sig = extracted.get('economic_signature', {}) or {}
        
        # If not in extracted_data, try direct access
        if not candidate_sig:
            candidate_sig = row_or_dict.get('economic_signature', {}) or {}
    elif isinstance(row_or_dict, dict):
        candidate_sig = row_or_dict.get('economic_signature', {}) or {}
    
    # If candidate doesn't have economic signature, try to infer from business description
    if not candidate_sig or not candidate_sig.get('capacity_unit') or candidate_sig.get('capacity_unit') == 'none':
        try:
            from features.archetype_inference import infer_archetype_fields_from_target
            # Create a dict with business_description for inference
            candidate_profile = {}
            if hasattr(row_or_dict, 'get'):
                candidate_profile['business_description'] = row_or_dict.get('business_description', '') or row_or_dict.get('business_activity', '')
                candidate_profile['business_activity'] = row_or_dict.get('business_activity', [])
                candidate_profile['revenue_model'] = row_or_dict.get('revenue_model', [])
            else:
                candidate_profile['business_description'] = row_or_dict.get('business_description', '') or row_or_dict.get('business_activity', '')
                candidate_profile['business_activity'] = row_or_dict.get('business_activity', [])
                candidate_profile['revenue_model'] = row_or_dict.get('revenue_model', [])
            
            inferred = infer_archetype_fields_from_target(candidate_profile)
            candidate_sig = {**candidate_sig, **inferred}
        except Exception:
            # If inference fails, cannot determine - gate not applicable
            return True
    
    # Infer candidate's economic engine
    candidate_engine = infer_economic_engine_from_signature(candidate_sig)
    if not candidate_engine:
        # Cannot determine candidate's economic engine - gate not applicable
        return True
    
    # Check if economic engines match
    return target_engine == candidate_engine

