"""
archetypes.py: Economic archetype classification and similarity matching.

This module enables comparing companies based on their economic structure
(capacity units, pricing basis, asset intensity, etc.) rather than just
customer segments or semantic similarity.

Key concepts:
- EconomicSignature: Captures HOW a company earns revenue
- Archetype: A template economic signature pattern
- Similarity: Distance-based comparison between signatures
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional


@dataclass
class EconomicSignature:
    """Economic signature capturing how a company earns revenue."""
    capacity_unit: str
    pricing_basis: List[str]
    asset_intensity_0_1: float
    revenue_recurring_0_1: float
    inventory_fragmentation_0_1: float
    demand_matching_role: str
    utilization_metric: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EconomicSignature":
        """Robust loading with defaults."""
        return cls(
            capacity_unit=d.get("capacity_unit", "none"),
            pricing_basis=d.get("pricing_basis", []) or [],
            asset_intensity_0_1=float(d.get("asset_intensity_0_1", d.get("asset_intensity", 0.0))),
            revenue_recurring_0_1=float(d.get("revenue_recurring_0_1", 0.0)),
            inventory_fragmentation_0_1=float(d.get("inventory_fragmentation_0_1", 0.0)),
            demand_matching_role=d.get("demand_matching_role", "none"),
            utilization_metric=d.get("utilization_metric", "none"),
        )


@dataclass
class Archetype:
    """An economic archetype template."""
    name: str
    sig: EconomicSignature


def _load_archetypes_config() -> Dict[str, Dict[str, Any]]:
    """
    Hardcoded archetype configurations.
    Can be replaced with YAML/config file later if needed.
    """
    return {
        # Pure consulting / advisory
        "consulting_services": {
            "capacity_unit": "hours",
            "pricing_basis": ["time_and_materials", "fixed_fee"],
            "asset_intensity_0_1": 0.15,
            "revenue_recurring_0_1": 0.4,
            "inventory_fragmentation_0_1": 0.0,
            "demand_matching_role": "none",
            "utilization_metric": "hours_utilized",
        },
        # Vacation rentals / hospitality platforms like Awaze, Airbnb, etc.
        "hospitality_rentals_aggregator": {
            "capacity_unit": "nights",
            "pricing_basis": ["ADR", "commission"],
            "asset_intensity_0_1": 0.3,
            "revenue_recurring_0_1": 0.5,
            "inventory_fragmentation_0_1": 0.8,  # many small units
            "demand_matching_role": "aggregator",
            "utilization_metric": "occupancy",
        },
        # Industrial OEM equipment makers like Husky, Kadant, Dover, etc.
        "industrial_oem_capital_goods": {
            "capacity_unit": "units_sold",
            "pricing_basis": ["product_sale", "time_and_materials"],
            "asset_intensity_0_1": 0.8,
            "revenue_recurring_0_1": 0.2,
            "inventory_fragmentation_0_1": 0.2,
            "demand_matching_role": "vertically_integrated",
            "utilization_metric": "throughput",
        },
        # Real estate / REIT-like rental yield
        "real_estate_rental_yield": {
            "capacity_unit": "square_feet",
            "pricing_basis": ["rent"],
            "asset_intensity_0_1": 0.95,
            "revenue_recurring_0_1": 0.8,
            "inventory_fragmentation_0_1": 0.4,
            "demand_matching_role": "vertically_integrated",
            "utilization_metric": "occupancy",
        },
    }


def load_archetypes() -> Dict[str, Archetype]:
    """Load all archetype templates."""
    raw = _load_archetypes_config()
    return {
        name: Archetype(name=name, sig=EconomicSignature.from_dict(cfg))
        for name, cfg in raw.items()
    }


def archetype_distance(a: EconomicSignature, b: EconomicSignature) -> float:
    """
    Compute distance between two economic signatures.
    Lower distance = more similar.
    """
    dist = 0.0

    # Numeric features (L1 distance)
    dist += abs(a.asset_intensity_0_1 - b.asset_intensity_0_1)
    dist += abs(a.revenue_recurring_0_1 - b.revenue_recurring_0_1)
    dist += abs(a.inventory_fragmentation_0_1 - b.inventory_fragmentation_0_1)

    # Enum-like fields: mismatch = 1, match = 0
    if a.capacity_unit != b.capacity_unit:
        dist += 1.0
    if a.demand_matching_role != b.demand_matching_role:
        dist += 1.0
    if a.utilization_metric != b.utilization_metric:
        dist += 1.0

    # Pricing basis: if there is no overlap, add penalty
    if not (set(a.pricing_basis) & set(b.pricing_basis)):
        dist += 1.0

    return dist


def classify_to_archetype(
    sig: EconomicSignature, archetypes: Dict[str, Archetype]
) -> Tuple[str, float]:
    """
    Classify a signature to the closest archetype.
    
    Returns:
        (best_archetype_name, distance)
    """
    best_name = "unknown"
    best_dist = float("inf")
    for name, arch in archetypes.items():
        d = archetype_distance(sig, arch.sig)
        if d < best_dist:
            best_name = name
            best_dist = d
    return best_name, best_dist


def similarity_from_distance(d: float, scale: float = 3.0) -> float:
    """
    Convert distance to [0,1] similarity.
    Using 1 / (1 + d/scale) gives smoother decay than 1/(1+d).
    """
    return 1.0 / (1.0 + (d / scale))


def pair_archetype_similarity(
    target_sig_dict: Dict[str, Any],
    candidate_sig_dict: Dict[str, Any],
    archetypes: Dict[str, Archetype],
) -> Dict[str, Any]:
    """
    Compute archetype similarity between target and candidate.
    
    Args:
        target_sig_dict: Economic signature dict from target
        candidate_sig_dict: Economic signature dict from candidate
        archetypes: Dict of archetype templates
    
    Returns:
        Dict with similarity, archetype classifications, and distances
    """
    if not target_sig_dict or not candidate_sig_dict:
        return {
            "similarity": 0.5,
            "target_archetype": "unknown",
            "candidate_archetype": "unknown",
            "target_distance": None,
            "candidate_distance": None,
        }

    target_sig = EconomicSignature.from_dict(target_sig_dict)
    cand_sig = EconomicSignature.from_dict(candidate_sig_dict)

    target_arch, d_t = classify_to_archetype(target_sig, archetypes)
    cand_arch, d_c = classify_to_archetype(cand_sig, archetypes)

    # If they land in different archetypes, we can treat that as a penalty.
    same_archetype = target_arch == cand_arch
    # For similarity we can use candidate distance to its archetype as proxy
    sim_arch = similarity_from_distance(d_c)
    if not same_archetype:
        # STRONG penalty for cross-archetype matches (e.g., REIT vs hospitality aggregator)
        # This is critical for filtering anti-comps
        sim_arch *= 0.2  # Strong penalty (was 0.5) - only 20% of similarity if different archetypes

    return {
        "similarity": sim_arch,
        "target_archetype": target_arch,
        "candidate_archetype": cand_arch,
        "target_distance": d_t,
        "candidate_distance": d_c,
    }

