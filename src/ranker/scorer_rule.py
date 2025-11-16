"""
scorer_rule.py: Rule-based scoring, threshold gates, percent contribution calc.
"""
import yaml
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '../../config/weights.yaml')


def load_weights_config(mode='all_segments'):
    """Load weights configuration."""
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, 'r') as f:
            config = yaml.safe_load(f)
            return config.get(mode, {})
    return {}


def rule_score(features, weights_config=None, mode='all_segments'):
    """
    Compute rule-based score and percent contributions.
    Returns: (score, pct_dict, passed_gates)
    """
    if weights_config is None:
        weights_config = load_weights_config(mode)
    
    weights = weights_config.get('weights', {})
    gates = weights_config.get('gates', {})
    
    # Extract feature values
    P = features.get('P', 0.0)
    C = features.get('C', 0.0)
    M = features.get('M', 0.5)  # Default to neutral if missing
    S = features.get('S', 0.0)
    I = features.get('I', 0.5)  # Default to neutral if missing
    E = features.get('E', 0.0)
    R = features.get('R', 0.5)  # Default to neutral if missing
    
    # Get weights
    w_P = weights.get('P', 0.28)
    w_C = weights.get('C', 0.28)
    w_M = weights.get('M', 0.18)
    w_S = weights.get('S', 0.16)
    w_I = weights.get('I', 0.06)
    w_E = weights.get('E', 0.03)
    w_R = weights.get('R', 0.01)
    
    # Compute weighted score
    score = 100 * (w_P * P + w_C * C + w_M * M + w_S * S + w_I * I + w_E * E + w_R * R)
    
    # Compute percent contributions
    contributions = {
        'P': w_P * P,
        'C': w_C * C,
        'M': w_M * M,
        'S': w_S * S,
        'I': w_I * I,
        'E': w_E * E,
        'R': w_R * R
    }
    total_contrib = sum(contributions.values())
    
    pct_dict = {}
    if total_contrib > 0:
        for key, contrib in contributions.items():
            pct_dict[f'pct_{key}'] = 100 * contrib / total_contrib
    else:
        for key in contributions.keys():
            pct_dict[f'pct_{key}'] = 0.0
    
    # Check gates
    passed_gates = True
    min_product_hits = gates.get('min_product_hits', 0)
    min_shared_segments = gates.get('min_shared_segments', 0)
    
    product_hits = features.get('product_hits', 0)
    customer_hits = features.get('customer_hits', 0)
    
    if product_hits < min_product_hits:
        passed_gates = False
    if customer_hits < min_shared_segments:
        passed_gates = False
    
    return score, pct_dict, passed_gates


if __name__ == "__main__":
    # Test
    features = {
        'P': 0.8,
        'C': 0.7,
        'M': 0.6,
        'S': 0.9,
        'I': 0.8,
        'E': 1.0,
        'R': 1.0,
        'product_hits': 3,
        'customer_hits': 2
    }
    score, pct, passed = rule_score(features)
    print(f"Score: {score:.2f}")
    print(f"Percent contributions: {pct}")
    print(f"Passed gates: {passed}")
