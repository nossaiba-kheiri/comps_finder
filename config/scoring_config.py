"""
scoring_config.py: Centralized configuration for scoring, gating, and penalties.
All thresholds and weights live here - no hardcoding in core logic.
"""
import os
import yaml

# Load config from YAML if exists, otherwise use defaults
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'scoring_config.yaml')

DEFAULT_SCORING_CONFIG = {
    # Core feature weights
    "weights": {
        "P": 0.35,
        "C": 0.30,
        "M": 0.20,
        "S": 0.15,
        # Optional: I, E, R (if needed)
        "I": 0.06,
        "E": 0.03,
        "R": 0.01,
    },
    
    # Business model gating
    "business_model": {
        # Which model types are simply incompatible with the target
        "disallowed_types": [
            "marketplace",
            "hardware",
            "financial_institution",
            "other",
        ],
        # Minimum share of services (0–1) for a candidate to be considered "services-heavy"
        "min_services_share_for_gate": 0.5,
        # How much to penalize if services_share is below an "anchor"
        "services_share_anchor": 0.5,
        "services_penalty_weight": 0.4,  # Strength of penalty (multiplier)
    },
    
    # Segment / customer overlap gating
    "segments": {
        # Minimum number of overlapping segments required for gate
        "min_shared_segments": 1,
        # Minimum customer similarity for gate (C feature, 0-1 scale)
        "min_customer_similarity": 0.4,
    },
    
    # Numeric gate on overall score (optional)
    "min_score_for_final_comp": 0.0,
    
    # Additional gates (legacy, for backward compatibility)
    "gates": {
        "min_product_hits": 0,  # Can be overridden by target.json or config
        "min_shared_segments": 1,  # Duplicate of segments.min_shared_segments (for compatibility)
    },
}


def load_scoring_config(config_path=None):
    """
    Load scoring configuration from YAML file, falling back to defaults.
    
    Args:
        config_path: Path to YAML config file (optional)
    
    Returns:
        dict: Scoring configuration
    """
    if config_path is None:
        config_path = CONFIG_PATH
    
    # Start with defaults
    config = DEFAULT_SCORING_CONFIG.copy()
    
    # Override with YAML if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Deep merge: update nested dicts
                    def deep_update(base, updates):
                        for key, value in updates.items():
                            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                                deep_update(base[key], value)
                            else:
                                base[key] = value
                    
                    deep_update(config, yaml_config)
        except Exception as e:
            print(f"Warning: Could not load scoring config from {config_path}: {e}")
            print("Using default configuration.")
    
    return config


# Global config instance (can be reloaded if needed)
SCORING_CONFIG = load_scoring_config()


def reload_config(config_path=None):
    """Reload configuration from file."""
    global SCORING_CONFIG
    SCORING_CONFIG = load_scoring_config(config_path)
    return SCORING_CONFIG

