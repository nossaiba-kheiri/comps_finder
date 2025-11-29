"""
business_model_config.py: Centralized configuration for business model classification and scoring.

All thresholds and weights live here - no hardcoding in core logic.
Can be loaded from JSON/YAML or environment variables.
"""
from dataclasses import dataclass
from typing import Set, Optional, Dict, Any
import os
import yaml
import sys

# Add config directory to path
config_dir = os.path.join(os.path.dirname(__file__), '../../config')
sys.path.insert(0, config_dir)


@dataclass
class BusinessModelConfig:
    """
    Configuration for business model classification and similarity scoring.
    
    All thresholds are parameterized here - no magic numbers in logic.
    """
    # Classification thresholds
    services_pure_min: float = 0.80  # ≥ 80% services ⇒ "services"
    software_pure_max: float = 0.20  # ≤ 20% services & has software ⇒ "software"
    
    hybrid_min: float = 0.40  # 40-60% services & has software ⇒ true balanced "hybrid_services_software"
    hybrid_max: float = 0.60
    
    services_led_min: float = 0.60  # > 60% services & has software ⇒ "services" (services-led hybrid)
    software_led_max: float = 0.40  # < 40% services & has software ⇒ "software" (software-led hybrid)
    
    # Business-model similarity scaling
    bm_scale_min: float = 0.4  # min multiplier when services_share very different
    bm_scale_max: float = 1.0  # max multiplier when identical
    
    # Hard gates for clearly non-comparable types
    gated_out_types: Set[str] = None  # e.g. {"hardware", "marketplace", "financial_institution"}
    
    # Business model similarity scoring weights (for B feature)
    similarity_share_weight: float = 0.6  # Weight for services share similarity
    similarity_type_weight: float = 0.3   # Weight for type bonus (identical/same family)
    similarity_capability_weight: float = 0.1  # Weight for required capabilities
    
    # Type bonus values
    type_bonus_identical: float = 0.5  # Bonus when business_model_type is identical
    type_bonus_same_family: float = 0.3  # Bonus when in same family (e.g., both services-heavy)
    type_bonus_none: float = 0.0  # No bonus
    
    # Required capabilities thresholds
    services_heavy_threshold: float = 0.6  # services_share >= this → requires services capabilities
    product_heavy_threshold: float = 0.4  # services_share <= this → requires product capabilities
    
    # Required capabilities penalty values
    missing_professional_services_penalty: float = 0.4
    missing_managed_services_penalty: float = 0.3
    missing_software_product_penalty: float = 0.4
    missing_hardware_product_penalty: float = 0.4
    
    # Validation thresholds (for LLM extraction validation)
    pure_software_max_services_share: float = 0.2  # Pure software should have services_share <= 0.2
    software_primary_max_services_share: float = 0.3  # Software-primary should have services_share <= 0.3
    pure_saas_services_share: float = 0.15  # Pure SaaS (only subscription_software revenue model)
    
    # Business model penalty (for scorer_rule.py)
    services_share_anchor_default: float = 0.5  # Default anchor when target not provided
    services_penalty_weight: float = 0.4  # Strength of penalty (multiplier)
    strong_penalty_for_failed_gate: float = 0.7  # Strong penalty when gate fails
    
    # Family groupings (for same-family logic)
    services_family: Set[str] = None  # Business model types that are "services-heavy"
    product_family: Set[str] = None   # Business model types that are "product-heavy"
    hardware_family: Set[str] = None
    marketplace_family: Set[str] = None
    
    def __post_init__(self):
        """Initialize sets with default values if None."""
        if self.gated_out_types is None:
            self.gated_out_types = {"hardware", "marketplace", "financial_institution"}
        
        if self.services_family is None:
            self.services_family = {"services", "hybrid_services_software"}
        
        if self.product_family is None:
            self.product_family = {"software", "hybrid_services_software"}
        
        if self.hardware_family is None:
            self.hardware_family = {"hardware"}
        
        if self.marketplace_family is None:
            self.marketplace_family = {"marketplace"}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BusinessModelConfig':
        """Create config from dictionary (e.g., from YAML/JSON)."""
        # Convert set-like lists to sets
        for key in ['gated_out_types', 'services_family', 'product_family', 
                    'hardware_family', 'marketplace_family']:
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = set(config_dict[key])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for YAML/JSON serialization)."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, set):
                result[key] = list(value)
            else:
                result[key] = value
        return result


def load_business_model_config(config_path: Optional[str] = None) -> BusinessModelConfig:
    """
    Load BusinessModelConfig from YAML file or create default.
    
    Args:
        config_path: Optional path to YAML config file. If None, tries to load from
                    scoring_config.yaml, then falls back to defaults.
    
    Returns:
        BusinessModelConfig instance
    """
    # Try to load from scoring_config.yaml first
    if config_path is None:
        config_path = os.path.join(config_dir, 'scoring_config.yaml')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract business_model_config section if it exists
            bm_config_dict = config_data.get('business_model_config', {})
            
            if bm_config_dict:
                return BusinessModelConfig.from_dict(bm_config_dict)
        except Exception as e:
            print(f"Warning: Failed to load business_model_config from {config_path}: {e}")
            print("Using default BusinessModelConfig")
    
    # Return default config
    return BusinessModelConfig()


# Global default instance
BM_CONFIG = load_business_model_config()


def classify_business_model(
    services_share: float,
    has_software_product: bool,
    cfg: Optional[BusinessModelConfig] = None,
) -> str:
    """
    Classify business model type based on services_share and software product flag.
    
    Returns: 'services' | 'software' | 'hybrid_services_software' | 'other'
    
    Args:
        services_share: Percentage of revenue from services (0.0-1.0)
        has_software_product: Whether company has software products
        cfg: Optional BusinessModelConfig (uses BM_CONFIG if not provided)
    
    Returns:
        str: Business model type
    """
    if cfg is None:
        cfg = BM_CONFIG
    
    # Clamp services_share to valid range
    services_share = max(0.0, min(1.0, services_share))
    
    # Pure-ish services
    if services_share >= cfg.services_pure_min:
        return "services"
    
    # Pure-ish software
    if has_software_product and services_share <= cfg.software_pure_max:
        return "software"
    
    # True balanced hybrid (both substantial)
    if has_software_product and cfg.hybrid_min <= services_share <= cfg.hybrid_max:
        return "hybrid_services_software"
    
    # Services-led with some software
    if has_software_product and services_share > cfg.services_led_min:
        return "services"
    
    # Software-led with some services
    if has_software_product and services_share < cfg.software_led_max:
        return "software"
    
    # Fallbacks when no software_product flag or ambiguous evidence
    if services_share > 0.5:
        return "services"
    elif has_software_product:
        return "software"
    else:
        return "other"


def business_model_similarity_scale(
    target_services_share: float,
    candidate_services_share: float,
    cfg: Optional[BusinessModelConfig] = None,
) -> float:
    """
    Compute a multiplicative scaling factor based on services_share similarity.
    
    Returns a factor in [bm_scale_min, bm_scale_max] based on how similar
    the services_share values are.
    
    Args:
        target_services_share: Target company's services share (0.0-1.0)
        candidate_services_share: Candidate company's services share (0.0-1.0)
        cfg: Optional BusinessModelConfig (uses BM_CONFIG if not provided)
    
    Returns:
        float: Scaling factor in [bm_scale_min, bm_scale_max]
    """
    if cfg is None:
        cfg = BM_CONFIG
    
    # Compute difference (0 = identical, 1 = completely different)
    diff = abs(target_services_share - candidate_services_share)
    
    # Convert to similarity (1 = identical, 0 = completely different)
    sim = 1.0 - diff
    
    # Map similarity [0, 1] → scale [bm_scale_min, bm_scale_max]
    span = cfg.bm_scale_max - cfg.bm_scale_min
    return cfg.bm_scale_min + span * sim

