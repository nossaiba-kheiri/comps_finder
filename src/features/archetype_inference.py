"""
archetype_inference.py: Infer NEW archetype fields (capacity_unit, pricing_basis, etc.)
from business description and existing target data.

This is needed because targets created before the NEW schema don't have these fields,
but we can infer them from business description, revenue model, and business model type.
"""
from typing import Dict, List, Any, Optional


def infer_archetype_fields_from_target(target: Dict) -> Dict[str, Any]:
    """
    Infer NEW archetype fields from target's business description and existing data.
    
    This allows targets created before the NEW schema to still work with archetype matching.
    
    Args:
        target: Target dict with business_description, business_model_type, revenue_model, etc.
    
    Returns:
        Dict with NEW archetype fields: capacity_unit, pricing_basis, revenue_recurring_0_1,
        inventory_fragmentation_0_1, demand_matching_role, utilization_metric
    """
    business_desc = (target.get('business_description', '') or '').lower()
    business_model = (target.get('business_model_type', '') or '').lower()
    revenue_model = target.get('revenue_model', []) or []
    revenue_model_lower = [str(rm).lower() for rm in revenue_model if rm]
    
    # Initialize with defaults
    capacity_unit = "none"
    pricing_basis = []
    revenue_recurring_0_1 = 0.0
    inventory_fragmentation_0_1 = 0.0
    demand_matching_role = "none"
    utilization_metric = "none"
    
    # HOSPITALITY / VACATION RENTALS (like Awaze, Airbnb)
    # Also includes hotels, resorts (same economic engine: revenue per night)
    if any(keyword in business_desc for keyword in [
        'vacation rental', 'holiday rental', 'short-term rental', 'booking platform',
        'hospitality platform', 'accommodation', 'nights', 'adr', 'revpar',
        'occupancy rate', 'property management', 'booking service',
        'hotel', 'hotels', 'resort', 'resorts', 'hospitality services', 'lodging'
    ]):
        capacity_unit = "nights"
        pricing_basis = ["ADR", "commission"]
        revenue_recurring_0_1 = 0.5  # Mix of recurring bookings and one-time
        inventory_fragmentation_0_1 = 0.8  # Many small properties
        demand_matching_role = "aggregator"  # Matches supply (properties) with demand (travelers)
        utilization_metric = "occupancy"
    
    # CONSULTING / PROFESSIONAL SERVICES
    elif any(keyword in business_desc for keyword in [
        'consulting', 'advisory', 'professional services', 'implementation',
        'transformation', 'strategic', 'project-based'
    ]) or business_model == 'services':
        capacity_unit = "hours"
        pricing_basis = ["time_and_materials", "fixed_fee"]
        revenue_recurring_0_1 = 0.4  # Mix of projects and retainers
        inventory_fragmentation_0_1 = 0.0  # No inventory
        demand_matching_role = "none"
        utilization_metric = "hours_utilized"
    
    # REAL ESTATE / REIT (rental yield, cap rate)
    elif any(keyword in business_desc for keyword in [
        'reit', 'real estate investment', 'property investment', 'rental yield',
        'cap rate', 'net lease', 'commercial property', 'multifamily property',
        'square feet', 'sq ft', 'leasing', 'property ownership'
    ]) or 'rent' in revenue_model_lower:
        capacity_unit = "square_feet"
        pricing_basis = ["rent"]
        revenue_recurring_0_1 = 0.8  # Mostly recurring rent
        inventory_fragmentation_0_1 = 0.4  # Some fragmentation (multiple properties)
        demand_matching_role = "vertically_integrated"  # Owns assets
        utilization_metric = "occupancy"
    
    # INDUSTRIAL / CAPITAL EQUIPMENT
    elif any(keyword in business_desc for keyword in [
        'equipment', 'machinery', 'manufacturing', 'industrial', 'capital goods',
        'injection molding', 'automation', 'systems', 'units sold'
    ]) or business_model == 'hardware':
        capacity_unit = "units_sold"
        pricing_basis = ["product_sale", "time_and_materials"]
        revenue_recurring_0_1 = 0.2  # Mostly one-time sales
        inventory_fragmentation_0_1 = 0.2  # Few product lines
        demand_matching_role = "vertically_integrated"
        utilization_metric = "throughput"
    
    # MARKETPLACE / TWO-SIDED PLATFORM
    elif any(keyword in business_desc for keyword in [
        'marketplace', 'platform', 'two-sided', 'transaction fee', 'commission',
        'connecting buyers and sellers', 'peer-to-peer'
    ]) or business_model == 'marketplace':
        capacity_unit = "none"  # Transactions, not capacity-based
        pricing_basis = ["commission", "transaction_fees"]
        revenue_recurring_0_1 = 0.3  # Transaction-based, not recurring
        inventory_fragmentation_0_1 = 0.9  # Many small transactions
        demand_matching_role = "marketplace"  # Matches supply and demand
        utilization_metric = "none"
    
    # SOFTWARE / SAAS
    elif any(keyword in business_desc for keyword in [
        'software', 'saas', 'subscription', 'platform', 'cloud', 'api'
    ]) or business_model == 'software' or 'subscription_software' in revenue_model_lower:
        capacity_unit = "none"  # Access-based, not capacity
        pricing_basis = ["subscription"]
        revenue_recurring_0_1 = 0.9  # Mostly recurring subscriptions
        inventory_fragmentation_0_1 = 0.0  # No inventory
        demand_matching_role = "none"
        utilization_metric = "none"
    
    # Default fallback: try to infer from revenue_model
    else:
        if 'rent' in revenue_model_lower:
            capacity_unit = "square_feet"
            pricing_basis = ["rent"]
            revenue_recurring_0_1 = 0.8
        elif 'subscription' in revenue_model_lower or 'recurring' in revenue_model_lower:
            capacity_unit = "none"
            pricing_basis = ["subscription"]
            revenue_recurring_0_1 = 0.7
        elif 'commission' in revenue_model_lower or 'transaction' in revenue_model_lower:
            capacity_unit = "none"
            pricing_basis = ["commission"]
            revenue_recurring_0_1 = 0.3
            demand_matching_role = "marketplace"
    
    return {
        "capacity_unit": capacity_unit,
        "pricing_basis": pricing_basis,
        "revenue_recurring_0_1": revenue_recurring_0_1,
        "inventory_fragmentation_0_1": inventory_fragmentation_0_1,
        "demand_matching_role": demand_matching_role,
        "utilization_metric": utilization_metric
    }

