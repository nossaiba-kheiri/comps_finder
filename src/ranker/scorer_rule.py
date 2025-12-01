"""
scorer_rule.py: Rule-based scoring, threshold gates, percent contribution calc.
All logic is configurable via scoring_config.py - no hardcoding.
"""
import numpy as np
import sys
import os

# Add config to path
config_path = os.path.join(os.path.dirname(__file__), '../../config')
sys.path.insert(0, config_path)

try:
    from scoring_config import SCORING_CONFIG, load_scoring_config
except ImportError:
    # Fallback if scoring_config.py not found
    print("Warning: scoring_config.py not found, using hardcoded defaults")
    SCORING_CONFIG = {
        "weights": {"P": 0.28, "C": 0.28, "M": 0.18, "S": 0.16, "I": 0.06, "E": 0.03, "R": 0.01},
        "business_model": {
            "disallowed_types": ["marketplace", "hardware", "financial_institution", "other"],
            "min_services_share_for_gate": 0.5,
            "services_share_anchor": 0.5,
            "services_penalty_weight": 0.4,
        },
        "segments": {
            "min_shared_segments": 1,
            "min_customer_similarity": 0.4,
        },
        "min_score_for_final_comp": 0.0,
    }
    def load_scoring_config(*args, **kwargs):
        return SCORING_CONFIG


def normalize_list(x):
    """Normalize list-like input to a list of lowercase strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x.strip().lower()]
    return [str(v).strip().lower() for v in x if v]


def compute_discipline_similarity(extracted_data, target_profile, config=None):
    """
    Compute discipline similarity (D) as a weighted combination of archetype, channel, and delivery similarities.
    This is used for soft discipline gating (quadratic penalty) instead of hard pass/fail gates.
    
    Args:
        extracted_data: Dict with revenue_archetypes, revenue_channels, delivery_modes, etc.
        target_profile: Target JSON dict (required for similarity computation)
        config: Optional scoring config dict (uses SCORING_CONFIG if not provided)
    
    Returns:
        float: Discipline similarity score D [0, 1], or 1.0 if computation fails
    """
    if config is None:
        config = SCORING_CONFIG
    
    if not target_profile:
        # If no target, return neutral score (no penalty)
        return 1.0
    
    try:
        from features.model_similarity import compute_model_similarities
        similarities = compute_model_similarities(extracted_data, target_profile)
        arch_sim = similarities.get('archetype_similarity', 0.0)
        chan_sim = similarities.get('channel_similarity', 0.0)
        delivery_sim = similarities.get('delivery_mode_similarity', 0.0)
        
        # Compute weighted discipline similarity (D)
        cfg = config.get("model_similarity", {})
        w_arch = cfg.get("discipline_archetype_weight", 0.50)
        w_chan = cfg.get("discipline_channel_weight", 0.30)
        w_del = cfg.get("discipline_delivery_weight", 0.20)
        
        # Normalize weights to sum to 1.0
        total_weight = w_arch + w_chan + w_del
        if total_weight > 0:
            w_arch /= total_weight
            w_chan /= total_weight
            w_del /= total_weight
        else:
            # Default equal weights if not configured
            w_arch = w_chan = w_del = 1.0 / 3.0
        
        D = w_arch * arch_sim + w_chan * chan_sim + w_del * delivery_sim
        
        return float(np.clip(D, 0.0, 1.0))
    except Exception as e:
        # If similarity computation fails, return neutral score (no penalty)
        print(f"  Warning: Discipline similarity computation failed: {e}, using neutral score")
        return 1.0


def compute_discipline_penalty(discipline_similarity, config=None):
    """
    Compute soft discipline penalty using quadratic formula.
    If D < MIN_D, apply penalty: score_discipline *= (D / MIN_D) ** 2
    
    Args:
        discipline_similarity: Discipline similarity score D [0, 1]
        config: Optional scoring config dict (uses SCORING_CONFIG if not provided)
    
    Returns:
        float: Penalty multiplier [0, 1] to apply to score
    """
    if config is None:
        config = SCORING_CONFIG
    
    cfg = config.get("model_similarity", {})
    MIN_D = cfg.get("min_discipline_similarity", 0.15)
    
    if discipline_similarity >= MIN_D:
        # No penalty if D >= MIN_D
        return 1.0
    
    # Quadratic penalty: (D / MIN_D) ** 2
    # If D is half the min (e.g., 0.07 vs 0.15) → penalty ≈ (0.07 / 0.15)^2 ≈ 0.22
    # If D is 0 → penalty = 0 (score goes to 0, but purely by distance, not explicit rules)
    penalty_multiplier = (discipline_similarity / MIN_D) ** 2 if MIN_D > 0 else 0.0
    
    return float(np.clip(penalty_multiplier, 0.0, 1.0))


def gate_model_similarity(extracted_data, config=None, target_profile=None):
    """
    Legacy gate function - kept for backward compatibility.
    Now returns True (always pass) since we use soft discipline penalties instead of hard gates.
    
    Args:
        extracted_data: Dict with revenue_archetypes, revenue_channels, delivery_modes, etc.
        config: Optional scoring config dict (uses SCORING_CONFIG if not provided)
        target_profile: Target JSON dict (required for similarity computation)
    
    Returns:
        bool: Always True (soft penalties handle filtering now)
    """
    # Always pass - soft discipline penalties handle filtering
    return True


# REMOVED: gate_business_model and gate_business_model_legacy
# Hard gates have been replaced by soft penalties:
# - Economic similarity penalty (kills SaaS vs consulting mismatches)
# - Discipline penalty (kills wrong archetype/channel/delivery mismatches)
# - Business model penalty (soft penalty based on services_share mismatch)
# 
# All companies are now ranked, but mismatches are heavily downweighted via penalties.
# This ensures we always have a full list of comparables, while still penalizing wrong matches.


def compute_segment_overlap(row_or_dict, target_profile):
    """
    Compute number of overlapping segments between candidate and target.
    Uses target's customer_segment (who they serve) and candidate's customer_segment.
    Generic - works with any target.json structure.
    
    Args:
        row_or_dict: Dict/Series with customer_segment or customer_segments
        target_profile: Target JSON dict with customer_segment
    
    Returns:
        int: Number of overlapping segments
    """
    # Get target segments from customer_segment (who target serves)
    target_segments = normalize_list(target_profile.get("customer_segment", []))
    
    # Get candidate segments from customer_segment or customer_segments
    if hasattr(row_or_dict, 'get'):
        # Dict/Series - check customer_segment first, then customer_segments
        candidate_segments = normalize_list(
            row_or_dict.get("customer_segment") or row_or_dict.get("customer_segments", [])
        )
    else:
        # List or other
        candidate_segments = normalize_list(row_or_dict)
    
    # Compute intersection (overlapping segments)
    target_set = set(target_segments)
    candidate_set = set(candidate_segments)
    
    return len(target_set & candidate_set)


def gate_hospitality_keywords(row_or_dict, target_profile, config=None):
    """
    Minimal keyword-based gate for hospitality/vacation rental targets.
    
    Checks if candidate has hospitality keywords in business_activity, customer_segment,
    or business_description. This is a simple but effective filter to exclude REITs,
    real estate investment companies, and other non-hospitality businesses.
    
    Args:
        row_or_dict: Dict/Series with business_activity, customer_segment, or business_description
        target_profile: Target JSON dict (used to determine if this gate should be active)
        config: Optional scoring config dict
    
    Returns:
        bool: True if passes gate (has keywords or gate not active), False otherwise
    """
    if config is None:
        config = SCORING_CONFIG
    
    # Check if this gate is enabled for the target
    # Only apply if target is hospitality/vacation rental related
    target_desc = (target_profile.get('business_description', '') or '').lower()
    target_activity = ' '.join([str(a).lower() for a in target_profile.get('business_activity', [])])
    
    # Check if target is hospitality-related
    hospitality_keywords = [
        'vacation rental', 'holiday rental', 'short-term rental', 'booking platform',
        'hospitality', 'hotel', 'resort', 'accommodation', 'lodging'
    ]
    is_hospitality_target = any(kw in target_desc or kw in target_activity for kw in hospitality_keywords)
    
    if not is_hospitality_target:
        # Gate not applicable - always pass
        return True
    
    # Get keywords from config (configurable, not hardcoded)
    cfg = config.get("hospitality_keywords", {})
    required_keywords = cfg.get("keywords", [
        "hotel", "hotels", "resort", "resorts", "vacation rental",
        "holiday park", "campground", "glamping", "campsite",
        "guest", "traveller", "tourist", "stay", "short stay",
        "lodging", "hospitality", "bookings", "reservations"
    ])
    
    # Get candidate text from multiple sources
    candidate_texts = []
    
    # Helper to extract text from various formats
    def extract_text(value):
        if not value:
            return ''
        if isinstance(value, list):
            return ' '.join([str(v).lower() for v in value if v])
        return str(value).lower()
    
    # From business_activity (can be list, comma-separated string, or in extracted_data)
    business_activity = row_or_dict.get('business_activity', '')
    if not business_activity:
        # Try extracted_data if available
        if hasattr(row_or_dict, 'get'):
            extracted = row_or_dict.get('extracted_data', {}) or {}
            business_activity = extracted.get('business_activity', [])
    candidate_texts.append(extract_text(business_activity))
    
    # From customer_segment (can be list, comma-separated string, or in extracted_data)
    customer_segment = row_or_dict.get('customer_segment', '')
    if not customer_segment:
        # Try extracted_data if available
        if hasattr(row_or_dict, 'get'):
            extracted = row_or_dict.get('extracted_data', {}) or {}
            customer_segment = extracted.get('customer_segment', [])
    candidate_texts.append(extract_text(customer_segment))
    
    # From business_description (if available)
    business_desc = row_or_dict.get('business_description', '')
    if not business_desc and hasattr(row_or_dict, 'get'):
        extracted = row_or_dict.get('extracted_data', {}) or {}
        business_desc = extracted.get('business_description', '')
    if business_desc:
        candidate_texts.append(str(business_desc).lower())
    
    # Combine all candidate text
    combined_text = ' '.join(candidate_texts).lower()
    
    # Check if any required keyword appears in candidate text
    has_keyword = any(keyword.lower() in combined_text for keyword in required_keywords)
    
    return has_keyword


def gate_segments(row_or_dict, target_profile, config=None, vocabulary=None):
    """
    Generic segment/customer overlap gate using continuous segment distribution vectors.
    
    Uses cosine similarity on segment distributions (portfolio exposure), not binary tag matching.
    
    SPECIAL LOGIC FOR DIVERSIFIED TARGETS:
    - If target is diversified (multi-segment), then candidates that are too concentrated
      (single-segment) should be gated out unless they match multiple segments.
    - If target is concentrated (single-segment), then any candidate with good segment
      similarity passes (no concentration penalty).
    
    Generic - works for any industry, any number of segments.
    
    Args:
        row_or_dict: Dict/Series with segment_mix, segments, or customer_segment
        target_profile: Target JSON dict
        config: Optional scoring config dict
        vocabulary: Optional segment vocabulary (for efficiency if precomputed)
    
    Returns:
        bool: True if passes gate, False otherwise
    """
    if config is None:
        config = SCORING_CONFIG
    
    cfg = config.get("segments", {})
    min_segment_similarity = cfg.get("min_segment_similarity", 0.3)  # Lowered default from 0.5 to 0.3 (more lenient)
    max_entropy_penalty_for_diversified = cfg.get("max_entropy_penalty_for_diversified", 0.6)  # Max allowed entropy penalty when target is diversified
    diversification_threshold = cfg.get("diversification_threshold", 0.3)  # Entropy threshold to consider target "diversified"
    
    try:
        # OPTIMIZATION: Reuse precomputed segment similarity result if available (avoid redundant computation)
        # This was computed once in compute_features(), stored in features_df, and passed here
        precomputed_seg_s_result = None
        if hasattr(row_or_dict, 'get'):
            precomputed_seg_s_result = row_or_dict.get('_seg_s_result')
        elif isinstance(row_or_dict, dict):
            precomputed_seg_s_result = row_or_dict.get('_seg_s_result')
        
        if precomputed_seg_s_result is not None:
            # Reuse precomputed result - no need to recompute!
            seg_s_result = precomputed_seg_s_result
            sim_cosine = seg_s_result.get('sim_cosine', 0.0)
            S = seg_s_result.get('S', 0.0)
            is_target_diversified = seg_s_result.get('is_target_diversified', False)
            penalty_entropy = seg_s_result.get('penalty_entropy', 0.0)
            entropy_candidate = seg_s_result.get('entropy_candidate', 0.0)
        else:
            # Fallback: compute if not precomputed (shouldn't happen in normal flow)
            from features.segment_distribution import compute_segment_s_score
            
            # Convert row_or_dict to dict if needed
            if hasattr(row_or_dict, 'to_dict'):
                candidate_data = row_or_dict.to_dict()
            elif hasattr(row_or_dict, 'get'):
                candidate_data = dict(row_or_dict)
            else:
                candidate_data = row_or_dict if isinstance(row_or_dict, dict) else {}
            
            lambda_entropy = cfg.get("concentration_penalty_weight", 0.4)
            seg_s_result = compute_segment_s_score(
                target_profile,
                candidate_data,
                vocabulary=vocabulary,
                lambda_entropy=lambda_entropy,
                diversification_threshold=diversification_threshold
            )
            sim_cosine = seg_s_result.get('sim_cosine', 0.0)
            S = seg_s_result.get('S', 0.0)
            is_target_diversified = seg_s_result.get('is_target_diversified', False)
            penalty_entropy = seg_s_result.get('penalty_entropy', 0.0)
            entropy_candidate = seg_s_result.get('entropy_candidate', 0.0)
        
        # Gate logic: Segment similarity (sim_cosine) is first-class
        # Must have sufficient segment similarity
        # BUT: If vocabulary is empty or very small, be more lenient (fallback to customer_segment check)
        has_sufficient_similarity = sim_cosine >= min_segment_similarity
        
        # If segment similarity is low but we have customer_segment overlap, be lenient
        # This handles cases where segment_mix extraction failed but customer_segment exists
        if not has_sufficient_similarity:
            # Check if we have customer_segment overlap as fallback
            if hasattr(row_or_dict, 'get'):
                customer_hits = row_or_dict.get('customer_hits', 0)
                segments_overlap = row_or_dict.get('segments_overlap', 0)
            elif isinstance(row_or_dict, dict):
                customer_hits = row_or_dict.get('customer_hits', 0)
                segments_overlap = row_or_dict.get('segments_overlap', 0)
            else:
                customer_hits = 0
                segments_overlap = 0
            
            # If we have customer overlap, be lenient (segment extraction might have failed)
            if customer_hits > 0 or segments_overlap > 0:
                # Lower threshold for fallback
                if sim_cosine >= (min_segment_similarity * 0.6):  # 60% of threshold
                    return True  # Pass with lower threshold if customer overlap exists
            
            return False  # Fail gate if segment similarity is too low and no customer overlap
        
        # SPECIAL LOGIC FOR DIVERSIFIED TARGETS:
        # If target is diversified, penalize/gate out candidates that are too concentrated
        # (single-segment candidates should be rejected unless they match multiple segments)
        if is_target_diversified:
            # Target is diversified - candidate must also be somewhat diversified OR have very high segment similarity
            # If candidate is too concentrated (low entropy), gate it out unless it matches multiple segments
            is_candidate_concentrated = entropy_candidate < diversification_threshold
            
            if is_candidate_concentrated:
                # Candidate is concentrated (single-segment) but target is diversified
                # Only allow if segment similarity is very high (candidate matches target's segments well)
                # OR if entropy penalty is not too severe
                min_similarity_for_concentrated = cfg.get("min_similarity_for_concentrated_candidate", 0.5)  # Lowered from 0.7 to 0.5 (more lenient)
                has_very_high_similarity = sim_cosine >= min_similarity_for_concentrated
                has_acceptable_penalty = penalty_entropy <= max_entropy_penalty_for_diversified
                
                # Gate out concentrated candidates unless they have very high similarity or acceptable penalty
                return has_very_high_similarity or has_acceptable_penalty
            else:
                # Candidate is also diversified - check if entropy penalty is acceptable
                return penalty_entropy <= max_entropy_penalty_for_diversified
        else:
            # Target is concentrated (single-segment) - no concentration penalty needed
            # Pure segment similarity is sufficient
            return True
        
    except Exception as e:
        # Fallback to legacy binary matching if vector-based approach fails
        print(f"  Warning: Segment distribution similarity failed: {e}, falling back to legacy gate")
        if hasattr(row_or_dict, 'get'):
            segments_overlap = row_or_dict.get('segments_overlap', None)
            if segments_overlap is None:
                segments_overlap = compute_segment_overlap(row_or_dict, target_profile)
            C = row_or_dict.get('C', 0.0)
        else:
            segments_overlap = compute_segment_overlap(row_or_dict, target_profile)
            C = 0.0
        
        min_shared_segments = cfg.get("min_shared_segments", 0)
        min_customer_similarity = cfg.get("min_customer_similarity", 0.2)  # Lowered from 0.3 to 0.2 (more lenient)
        has_segment_overlap = segments_overlap >= max(1, min_shared_segments)
        has_customer_similarity = C >= min_customer_similarity
        return has_segment_overlap or has_customer_similarity


def compute_base_score(features, config=None):
    """
    Compute base linear score from P, C, S, B features (and optionally I, E, R).
    Generic - driven by config weights.
    
    Args:
        features: Dict with P, C, S, B (and optionally I, E, R)
        config: Optional scoring config dict
    
    Returns:
        float: Base linear score (0-1 scale)
    """
    if config is None:
        config = SCORING_CONFIG
    
    weights = config.get("weights", {})
    
    # Extract feature values (default to 0.0 for missing, 0.5 for neutral features)
    # M removed - redundant with S (both measure segment similarity)
    P = float(features.get('P', 0.0))
    C = float(features.get('C', 0.0))
    S = float(features.get('S', 0.0))
    B = float(features.get('B', 0.0))  # Legacy: Business model similarity
    V = float(features.get('V', 0.0))  # Vertical similarity (multi-hot encoding)
    E_SIG = float(features.get('E_SIG', 0.0))  # NEW: Economic signature similarity (universal economic structure matching)
    I = float(features.get('I', 0.5))  # Default to neutral if missing
    E = float(features.get('E', 0.0))
    R = float(features.get('R', 0.5))  # Default to neutral if missing
    
    # Get weights (with defaults)
    w_P = weights.get('P', 0.15)
    w_C = weights.get('C', 0.10)
    w_S = weights.get('S', 0.10)
    w_B = weights.get('B', 0.10)  # Legacy business model similarity weight
    w_V = weights.get('V', 0.10)  # Vertical similarity weight
    w_E_SIG = weights.get('E_SIG', 0.40)  # Economic signature similarity - PRIMARY FEATURE
    w_I = weights.get('I', 0.03)
    w_E = weights.get('E', 0.01)
    w_R = weights.get('R', 0.01)
    
    # SIC industry bonus (small hint, not a filter)
    # Treat same SIC/industry as a small bonus (0.05 weight)
    same_sic = float(features.get('same_sic', 0.0))
    w_sic_bonus = 0.05  # Small weight - don't want to lose cross-SIC comps
    
    # Compute weighted score (includes E_SIG as primary feature)
    score_linear = (
        w_P * P +
        w_C * C +
        w_S * S +
        w_B * B +  # Legacy business model similarity
        w_V * V +  # Vertical similarity
        w_E_SIG * E_SIG +  # NEW: Economic signature similarity (universal matching)
        w_I * I +
        w_E * E +
        w_R * R +
        w_sic_bonus * same_sic  # Small SIC bonus
    )
    
    return float(np.clip(score_linear, 0.0, 1.0))


def apply_business_model_penalty(score_linear, extracted_data, config=None, target_profile=None, gate_passed=False):
    """
    Apply penalty based on services share and business model mismatch.
    Generic - driven by config and target profile.
    Uses BusinessModelConfig for all thresholds (no hardcoded values).
    
    If gate failed, apply STRONG penalty to ensure they don't rank high.
    Otherwise, apply soft penalty based on services share mismatch.
    
    Args:
        score_linear: Base linear score (0-1 scale)
        extracted_data: Dict with services_share_estimate, business_model_type
        config: Optional scoring config dict
        target_profile: Optional target JSON dict (used to dynamically set anchor)
        gate_passed: Whether the business model gate passed (if False, apply strong penalty)
    
    Returns:
        tuple: (penalty_value, adjusted_score)
            - penalty_value: The penalty applied (0-1 scale)
            - adjusted_score: score_linear - penalty (0-1 scale)
    """
    # Try to load BusinessModelConfig for penalty thresholds
    try:
        import sys
        import os
        config_path = os.path.join(os.path.dirname(__file__), '../config')
        sys.path.insert(0, config_path)
        from config.business_model_config import BM_CONFIG
        bm_cfg = BM_CONFIG
    except ImportError:
        # Fallback to scoring config
        bm_cfg = None
    
    if config is None:
        config = SCORING_CONFIG
    
    cfg = config.get("business_model", {})
    
    # If gate failed, apply STRONG penalty (use config value)
    if not gate_passed:
        if bm_cfg:
            strong_penalty = bm_cfg.strong_penalty_for_failed_gate
        else:
            strong_penalty = cfg.get("strong_penalty_for_failed_gate", 0.7)
        adjusted_score = max(0.0, score_linear - strong_penalty)
        return strong_penalty, adjusted_score
    
    # Otherwise, apply soft penalty based on services share mismatch
    # Dynamically set anchor based on target profile
    if target_profile:
        anchor = float(target_profile.get('services_share_estimate', 0.5) or 0.5)
    else:
        if bm_cfg:
            anchor = bm_cfg.services_share_anchor_default
        else:
            anchor = cfg.get("services_share_anchor", 0.5)
    
    if bm_cfg:
        w_pen = bm_cfg.services_penalty_weight
    else:
        w_pen = cfg.get("services_penalty_weight", 0.4)
    
    # Get services share (default to 0 if missing - conservative)
    services_share = extracted_data.get('services_share_estimate', 0.0) if extracted_data else 0.0
    services_share = float(services_share or 0.0)
    
    # Penalty is positive when services_share < anchor; 0 otherwise
    # Clipped to [0, 1] to ensure penalty is bounded
    penalty_raw = np.clip(anchor - services_share, 0.0, anchor)
    penalty_value = w_pen * penalty_raw
    
    # Subtract penalty from linear score
    adjusted_score = max(0.0, score_linear - penalty_value)  # Don't go negative
    
    return penalty_value, adjusted_score


def rule_score(features, target_profile=None, extracted_data=None, config=None):
    """
    Compute rule-based score with business model penalty and gates.
    Generic - driven by config and target.json.
    
    Args:
        features: Dict with P, C, M, S, I, E, R, product_hits, customer_hits
        target_profile: Target JSON dict (optional, for segment gates)
        extracted_data: Dict with business_model_type, services_share_estimate (optional)
        config: Optional scoring config dict
    
    Returns:
        tuple: (score_100, pct_dict, passed_gates, gate_details)
            - score_100: Final score on 0-100 scale
            - pct_dict: Percent contributions by feature
            - passed_gates: bool (all gates passed)
            - gate_details: dict with individual gate results
    """
    if config is None:
        config = SCORING_CONFIG
    
    # 1. Compute base linear score
    score_linear = compute_base_score(features, config)
    
    # 1.5. Apply INTERACTION PENALTY (similarity scale + P*C cross-term)
    # Combine two effects:
    # 1. Similarity scale based on max(C, segment_similarity) - downweights if no customer/segment match
    # 2. P*C interaction term - multiplicative cross-term (P and C must both be high)
    P = float(features.get('P', 0.0))  # Product overlap
    C = float(features.get('C', 0.0))  # Customer overlap
    segment_similarity = float(features.get('segment_similarity', 0.0))  # Segment distribution similarity
    # Fallback to sim_cosine if segment_similarity not available
    if segment_similarity == 0.0:
        segment_similarity = float(features.get('sim_cosine', 0.0))
    
    # Similarity scale: ranges from 0.3 (no match) to 1.0 (strong match)
    # Based on max(C, segment_similarity) - if either is high, we keep weight
    similarity_scale = 0.3 + 0.7 * max(C, segment_similarity)
    similarity_scale = np.clip(similarity_scale, 0.3, 1.0)  # Ensure it's in [0.3, 1.0] range
    
    # Interaction term: multiplicative P*C cross-term
    # When P*C = 0 (either P=0 or C=0), interaction_scale = 0.3 (downweighted)
    # When P*C = 1 (both P=1 and C=1), interaction_scale = 1.0 (full weight)
    interaction_term = P * C  # Multiplicative cross-term
    interaction_scale = 0.3 + 0.7 * interaction_term  # Scale from 0.3 to 1.0 based on P*C
    interaction_scale = np.clip(interaction_scale, 0.3, 1.0)  # Ensure it's in [0.3, 1.0] range
    
    # Apply both scales: similarity_scale (customer/segment match) AND interaction_scale (P*C)
    # Combined effect: ADDITIVE instead of multiplicative (average of both scales)
    # This is less harsh than multiplicative - if one is high, it helps the score
    combined_scale = (similarity_scale + interaction_scale) / 2.0  # Average of both scales
    combined_scale = np.clip(combined_scale, 0.0, 1.0)  # Ensure it's in [0.0, 1.0] range
    
    # Apply combined_scale ADDITIVELY (linear adjustment) instead of multiplicatively
    # Instead of: score_linear = score_linear * combined_scale
    # Use: score_linear = score_linear + (combined_scale - 1.0) * adjustment_weight
    # This applies a linear penalty/bonus based on how far combined_scale is from 1.0
    adjustment_weight = 0.5  # How much to adjust (0.5 = moderate adjustment)
    score_adjustment = (combined_scale - 1.0) * adjustment_weight
    score_linear = score_linear + score_adjustment
    score_linear = np.clip(score_linear, 0.0, 1.0)  # Ensure score stays in [0, 1] range
    
    # 2. Add model similarity bonus (3-layer model)
    model_sim_cfg = config.get("model_similarity", {})
    w_arch_sim = model_sim_cfg.get("archetype_similarity_weight", 0.0)
    w_chan_sim = model_sim_cfg.get("channel_similarity_weight", 0.0)
    w_del_sim = model_sim_cfg.get("delivery_similarity_weight", 0.0)
    
    # Backward compatibility
    w_rev_sim = model_sim_cfg.get("revenue_similarity_weight", 0.0)
    
    similarity_bonus = 0.0
    if target_profile and extracted_data:
        try:
            from features.model_similarity import compute_model_similarities
            similarities = compute_model_similarities(extracted_data, target_profile)
            arch_sim = similarities.get('archetype_similarity', 0.0)
            chan_sim = similarities.get('channel_similarity', 0.0)
            delivery_sim = similarities.get('delivery_mode_similarity', 0.0)
            revenue_sim = similarities.get('revenue_model_similarity', chan_sim)  # Backward compat
            
            # Use 3-layer weights if available, otherwise fall back to legacy
            if w_arch_sim > 0 or w_chan_sim > 0:
                similarity_bonus = w_arch_sim * arch_sim + w_chan_sim * chan_sim + w_del_sim * delivery_sim
            else:
                similarity_bonus = w_rev_sim * revenue_sim + w_del_sim * delivery_sim
        except Exception:
            # If similarity computation fails, continue without bonus
            pass
    
    score_with_similarity = score_linear + similarity_bonus
    
    # 3. Apply segment concentration penalty (entropy-based)
    # Penalize companies that are too concentrated in a single segment
    # This is generic - works for any industry
    concentration_penalty_value = 0.0
    if target_profile and extracted_data:
        try:
            from features.segment_distribution import compute_segment_similarity
            # Get vocabulary from features if precomputed (for efficiency)
            vocabulary = features.get('segment_vocabulary', None)
            seg_sim_result = compute_segment_similarity(
                target_profile,
                extracted_data,
                vocabulary=vocabulary
            )
            concentration_penalty = seg_sim_result.get('concentration_penalty', 0.0)
            # Apply penalty: if concentration_penalty is high (single segment), reduce score
            # Lambda (λ) from config - how much to penalize concentration
            lambda_concentration = config.get("segments", {}).get("concentration_penalty_weight", 0.2)
            concentration_penalty_value = lambda_concentration * concentration_penalty
        except Exception:
            # If segment similarity computation fails, continue without penalty
            pass
    
    score_with_segment_penalty = max(0.0, score_with_similarity - concentration_penalty_value)
    
    # 4. Apply SOFT DISCIPLINE PENALTY (replaces hard gates)
    # Compute discipline similarity (D) and apply quadratic penalty if D < MIN_D
    discipline_penalty_multiplier = 1.0
    if target_profile and extracted_data:
        try:
            D = compute_discipline_similarity(extracted_data, target_profile, config)
            discipline_penalty_multiplier = compute_discipline_penalty(D, config)
            # Apply penalty: multiply score by penalty multiplier
            # If D < MIN_D, this heavily downweights the score (quadratic penalty)
            # If D = 0, score goes to 0 (but purely by distance, not explicit rules)
        except Exception as e:
            # If discipline computation fails, use neutral multiplier (no penalty)
            print(f"  Warning: Discipline penalty computation failed: {e}, using neutral multiplier")
            discipline_penalty_multiplier = 1.0
    
    score_with_discipline = score_with_segment_penalty * discipline_penalty_multiplier
    
    # 5. Apply ECONOMIC SIMILARITY PENALTY (new - kills SaaS vs consulting mismatch)
    # This is the key fix: compare economic mode, deal structure, buyer persona, transformation intent
    economic_penalty_multiplier = 1.0
    if target_profile and extracted_data:
        try:
            from features.economic_similarity import compute_economic_similarities
            economic_sims = compute_economic_similarities(extracted_data, target_profile)
            overall_economic_sim = economic_sims.get('overall_economic_similarity', 1.0)
            
            # Apply quadratic penalty if economic similarity is too low
            # MIN_ECON_SIM = 0.30 (stricter - kills SaaS vs consulting mismatches)
            MIN_ECON_SIM = config.get("model_similarity", {}).get("min_economic_similarity", 0.30)
            
            if overall_economic_sim < MIN_ECON_SIM:
                # Quadratic penalty: (sim / MIN) ** 2
                economic_penalty_multiplier = (overall_economic_sim / MIN_ECON_SIM) ** 2 if MIN_ECON_SIM > 0 else 0.0
            else:
                economic_penalty_multiplier = 1.0
        except Exception as e:
            # If economic similarity computation fails, use neutral multiplier (no penalty)
            print(f"  Warning: Economic similarity computation failed: {e}, using neutral multiplier")
            economic_penalty_multiplier = 1.0
    
    score_with_economic = score_with_discipline * economic_penalty_multiplier
    
    # 6. Business model gate removed - soft penalties handle filtering now
    # All companies are ranked, but mismatches are heavily downweighted via penalties
    gate_bm_passed = True  # Always pass (for reporting only - soft penalties handle filtering)
    
    # 7. Apply business model penalty (soft penalty, not a gate)
    penalty_value, score_adjusted = apply_business_model_penalty(
        score_with_economic, extracted_data, config, target_profile, gate_passed=True  # Always pass now (soft penalties handle it)
    )
    
    # 3. Convert to 0-100 scale
    score_100 = 100 * score_adjusted
    
    # 4. Compute percent contributions
    weights = config.get("weights", {})
    w_P = weights.get('P', 0.35)
    w_C = weights.get('C', 0.30)
    w_M = weights.get('M', 0.20)
    w_S = weights.get('S', 0.15)
    w_I = weights.get('I', 0.06)
    w_E = weights.get('E', 0.03)
    w_R = weights.get('R', 0.01)
    
    contributions = {
        'P': w_P * features.get('P', 0.0),
        'C': w_C * features.get('C', 0.0),
        'M': w_M * features.get('M', 0.5),
        'S': w_S * features.get('S', 0.0),
        'I': w_I * features.get('I', 0.5),
        'E': w_E * features.get('E', 0.0),
        'R': w_R * features.get('R', 0.5),
        'business_model_penalty': -penalty_value  # Negative contribution
    }
    
    total_contrib = sum(abs(v) for v in contributions.values()) if contributions else 1.0
    pct_dict = {}
    if total_contrib > 0:
        for key, contrib in contributions.items():
            pct_dict[f'pct_{key}'] = 100 * contrib / total_contrib
    else:
        for key in contributions.keys():
            pct_dict[f'pct_{key}'] = 0.0
    
    # 7. Check gates (for reporting/debugging, but not used for filtering - soft penalties handle it)
    gate_details = {}
    
    # Business model gate removed - soft penalties handle filtering
    gate_details['business_model'] = True  # Always pass (for reporting only - soft penalties handle filtering)
    
    # Segment gate (for reporting only - soft penalties handle filtering)
    if target_profile:
        # Create a combined dict for gate_segments
        row_for_gates = {**features}
        if extracted_data:
            row_for_gates.update(extracted_data)
        gate_details['segments'] = True  # Always pass now (soft penalties handle filtering)
        
        # NEW: Hospitality keywords gate (minimal rule-based filter)
        # This is a HARD gate - filters out REITs and non-hospitality companies
        gate_details['hospitality_keywords'] = gate_hospitality_keywords(
            row_for_gates, target_profile, config
        )
        
        # NEW: Economic engine gate (ensures same revenue mechanism)
        # This is a HARD gate - filters out companies with different economic engines
        # e.g., Awaze (revenue per night) vs REITs (revenue per square foot)
        try:
            from ranker.gate_economic_engine import gate_economic_engine
            gate_details['economic_engine'] = gate_economic_engine(
                row_for_gates, target_profile, config
            )
        except Exception as e:
            # If gate fails, default to pass (don't break pipeline)
            import warnings
            warnings.warn(f"Economic engine gate failed: {e}, defaulting to pass", stacklevel=2)
            gate_details['economic_engine'] = True
    else:
        gate_details['segments'] = True  # Pass if no target_profile provided
        gate_details['hospitality_keywords'] = True  # Pass if no target_profile
        gate_details['economic_engine'] = True  # Pass if no target_profile
    
    # Legacy gates (product_hits, customer_hits)
    gates = config.get('gates', {})
    min_product_hits = gates.get('min_product_hits', 0)
    min_shared_segments_legacy = gates.get('min_shared_segments', 0)
    
    product_hits = features.get('product_hits', 0)
    customer_hits = features.get('customer_hits', 0)
    
    gate_details['product_hits'] = product_hits >= min_product_hits
    gate_details['customer_hits'] = customer_hits >= min_shared_segments_legacy
    
    # Store discipline and economic similarity for debugging/explainability
    if target_profile and extracted_data:
        try:
            D = compute_discipline_similarity(extracted_data, target_profile, config)
            discipline_penalty = compute_discipline_penalty(D, config)
            gate_details['discipline_similarity'] = D
            gate_details['discipline_penalty_multiplier'] = discipline_penalty
            
            # Also store economic similarities
            from features.economic_similarity import compute_economic_similarities
            economic_sims = compute_economic_similarities(extracted_data, target_profile)
            gate_details['economic_mode_similarity'] = economic_sims.get('economic_mode_similarity', 1.0)
            gate_details['deal_structure_similarity'] = economic_sims.get('deal_structure_similarity', 1.0)
            gate_details['buyer_persona_similarity'] = economic_sims.get('buyer_persona_similarity', 1.0)
            gate_details['transformation_intent_similarity'] = economic_sims.get('transformation_intent_similarity', 1.0)
            gate_details['overall_economic_similarity'] = economic_sims.get('overall_economic_similarity', 1.0)
            gate_details['economic_penalty_multiplier'] = economic_penalty_multiplier
        except Exception:
            gate_details['discipline_similarity'] = 1.0
            gate_details['discipline_penalty_multiplier'] = 1.0
            gate_details['overall_economic_similarity'] = 1.0
            gate_details['economic_penalty_multiplier'] = 1.0
    
    # Store similarity scale and interaction term for debugging/explainability
    gate_details['similarity_scale'] = similarity_scale  # Based on max(C, segment_similarity)
    gate_details['interaction_scale'] = interaction_scale  # Based on P*C
    gate_details['combined_scale'] = combined_scale  # similarity_scale * interaction_scale
    gate_details['interaction_term_P_C'] = interaction_term  # P * C multiplicative cross-term
    gate_details['similarity_scale_P'] = P
    gate_details['similarity_scale_C'] = C
    gate_details['similarity_scale_segment'] = segment_similarity
    
    # All gates must pass (soft penalties handle filtering)
    # Check if all gates passed (including hospitality_keywords and economic_engine gates)
    passed_gates = all([
        gate_details.get('business_model', True),
        gate_details.get('segments', True),
        gate_details.get('hospitality_keywords', True),  # NEW: Must pass hospitality keywords gate
        gate_details.get('economic_engine', True),  # NEW: Must pass economic engine gate
        gate_details.get('product_hits', True),
        gate_details.get('customer_hits', True)
    ])
    
    # Return score_adjusted (0-1 scale) for ranking, in addition to score_100 (0-100 scale)
    return score_100, pct_dict, passed_gates, gate_details, score_adjusted


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
