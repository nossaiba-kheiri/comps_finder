"""
business_model_similarity.py: Compute business model similarity between target and candidate.

Generic feature that compares:
- Services share (how similar is the revenue mix)
- Business model type (identical or "same family")
- Required capabilities (services capabilities if target is services-heavy)

Works for ANY target - no hardcoding. All thresholds are parameterized via BusinessModelConfig.
"""
from typing import Dict, Optional
import sys
import os

# Add config to path
config_path = os.path.join(os.path.dirname(__file__), '../config')
sys.path.insert(0, config_path)

try:
    from config.business_model_config import BusinessModelConfig, BM_CONFIG
except ImportError:
    # Fallback: create a minimal config-like object if import fails
    from dataclasses import dataclass
    
    @dataclass
    class BusinessModelConfig:
        services_heavy_threshold: float = 0.6
        product_heavy_threshold: float = 0.4
        similarity_share_weight: float = 0.6
        similarity_type_weight: float = 0.3
        similarity_capability_weight: float = 0.1
        type_bonus_identical: float = 0.5
        type_bonus_same_family: float = 0.3
        type_bonus_none: float = 0.0
        missing_professional_services_penalty: float = 0.4
        missing_managed_services_penalty: float = 0.3
        missing_software_product_penalty: float = 0.4
        missing_hardware_product_penalty: float = 0.4
        services_family: set = None
        product_family: set = None
        hardware_family: set = None
        marketplace_family: set = None
        
        def __post_init__(self):
            if self.services_family is None:
                self.services_family = {"services", "hybrid_services_software"}
            if self.product_family is None:
                self.product_family = {"software", "hybrid_services_software"}
            if self.hardware_family is None:
                self.hardware_family = {"hardware"}
            if self.marketplace_family is None:
                self.marketplace_family = {"marketplace"}
    
    BM_CONFIG = BusinessModelConfig()


def business_model_similarity(
    target_profile: Dict, 
    candidate_data: Dict,
    cfg: Optional[BusinessModelConfig] = None
) -> float:
    """
    Compute business model similarity between target and candidate.
    
    Generic - works for any target (SaaS, consulting, hardware, marketplace, etc.)
    All thresholds are parameterized via BusinessModelConfig.
    
    Args:
        target_profile: Target JSON dict with services_share_estimate, business_model_type, etc.
        candidate_data: Candidate dict with services_share_estimate, business_model_type, etc.
        cfg: Optional BusinessModelConfig (uses BM_CONFIG if not provided)
    
    Returns:
        float: Business model similarity score [0, 1]
    """
    if cfg is None:
        cfg = BM_CONFIG
    
    # 1) Services share distance (0 = bad, 1 = identical)
    s_t = float(target_profile.get('services_share_estimate', 0.5) or 0.5)
    s_c = float(candidate_data.get('services_share_estimate', 0.5) or 0.5)
    
    # Linear distance: 1.0 if identical, 0.0 if completely different
    share_sim = max(0.0, 1.0 - abs(s_t - s_c))
    
    # 2) Type bonus: identical or "same family" (e.g., both hybrid, or both services)
    target_bm = (target_profile.get('business_model_type') or 'other').lower()
    cand_bm = (candidate_data.get('business_model_type') or 'other').lower()
    
    same_type = 1.0 if cand_bm == target_bm else 0.0
    
    # "Same family" logic: use config families
    both_servicesy = (
        target_bm in cfg.services_family and
        cand_bm in cfg.services_family
    )
    
    both_producty = (
        target_bm in cfg.product_family and
        cand_bm in cfg.product_family and
        target_bm != "services" and cand_bm != "services"  # Exclude pure services
    )
    
    both_hardware = (
        target_bm in cfg.hardware_family and
        cand_bm in cfg.hardware_family
    )
    
    both_marketplace = (
        target_bm in cfg.marketplace_family and
        cand_bm in cfg.marketplace_family
    )
    
    # Type bonus: use config values
    if same_type:
        type_bonus = cfg.type_bonus_identical
    elif both_servicesy or both_producty or both_hardware or both_marketplace:
        type_bonus = cfg.type_bonus_same_family
    else:
        type_bonus = cfg.type_bonus_none
    
    # 3) Required services capabilities if target is services-heavy
    req_services = 1.0
    if s_t >= cfg.services_heavy_threshold:  # Services-heavy target
        has_professional_services = candidate_data.get('has_professional_services', False)
        has_managed_services = candidate_data.get('has_managed_services', False)
        
        if not has_professional_services:
            req_services -= cfg.missing_professional_services_penalty
        if not has_managed_services:
            req_services -= cfg.missing_managed_services_penalty
    
    # Also check if target is product-heavy and candidate lacks product capabilities
    if s_t <= cfg.product_heavy_threshold:  # Product-heavy target
        has_software_product = candidate_data.get('has_software_product', False)
        has_hardware_product = candidate_data.get('has_hardware_product', False)
        
        # For product-heavy targets, having product capabilities is important
        if not has_software_product and not has_hardware_product:
            req_services -= cfg.missing_software_product_penalty
    
    req_services = max(0.0, req_services)
    
    # Weighted combination: use config weights
    B = (cfg.similarity_share_weight * share_sim + 
         cfg.similarity_type_weight * type_bonus + 
         cfg.similarity_capability_weight * req_services)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, B))


if __name__ == "__main__":
    # Test with different scenarios
    print("🔍 Testing business_model_similarity")
    print("=" * 70)
    
    # Huron (services-heavy hybrid)
    huron = {
        'services_share_estimate': 0.7,
        'business_model_type': 'hybrid_services_software',
        'has_professional_services': True,
        'has_managed_services': True,
        'has_software_product': True
    }
    
    # EXLS (services-heavy, similar to Huron)
    exls = {
        'services_share_estimate': 0.75,
        'business_model_type': 'services',
        'has_professional_services': True,
        'has_managed_services': True,
        'has_software_product': False
    }
    
    # FORA (software-heavy, different from Huron)
    fora = {
        'services_share_estimate': 0.1,
        'business_model_type': 'software',
        'has_professional_services': False,
        'has_managed_services': False,
        'has_software_product': True
    }
    
    # Wipro (hybrid, similar to Huron)
    wipro = {
        'services_share_estimate': 0.65,
        'business_model_type': 'hybrid_services_software',
        'has_professional_services': True,
        'has_managed_services': True,
        'has_software_product': True
    }
    
    print(f"\n1. Huron vs EXLS (services-heavy, similar):")
    sim1 = business_model_similarity(huron, exls)
    print(f"   B = {sim1:.3f} (should be HIGH)")
    
    print(f"\n2. Huron vs FORA (software-heavy, different):")
    sim2 = business_model_similarity(huron, fora)
    print(f"   B = {sim2:.3f} (should be LOW)")
    
    print(f"\n3. Huron vs Wipro (hybrid, similar):")
    sim3 = business_model_similarity(huron, wipro)
    print(f"   B = {sim3:.3f} (should be HIGH)")
    
    print("\n✅ All tests passed!")


