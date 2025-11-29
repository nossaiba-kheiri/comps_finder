"""
economic_similarity.py: Compute economic mode, deal structure, buyer persona, and transformation intent similarity.

These vectors capture the economic engine differences that distinguish companies GENERICALLY:
- Services firms (hourly billable, fixed-fee projects, MSA multiyear) vs Product firms (subscription software, per-seat)
- Transformation partners (strategic buyers) vs Product buyers (operational buyers)
- Strategic transformation vs automation/workflow products

Works for ANY industry - no hardcoding. Examples:
- Consulting vs SaaS (any industry)
- BPO vs Consulting (any industry)
- Hardware vs Software (any industry)
- Marketplace vs Services (any industry)
"""
import numpy as np
from typing import Dict, Optional, List


# Economic mode dimensions (delivery economics)
ECONOMIC_MODE_DIMENSIONS = [
    "hourly_billable",
    "fixed_fee_projects",
    "managed_services",
    "subscription_software",
    "performance_based",
    "risk_bearing"
]

# Deal structure dimensions
DEAL_STRUCTURE_DIMENSIONS = [
    "MSA_multiyear",
    "project_based_SOW",
    "platform_license",
    "per_member_per_month",
    "per_provider",
    "per_seat_saas",
    "transaction_based"
]

# Buyer persona dimensions
BUYER_PERSONA_DIMENSIONS = [
    "CIO",
    "CFO",
    "provost",
    "hospital_COO",
    "network_manager",
    "clinic_owner",
    "IT_system_admin",
    "claims_adjuster",
    "EMR_billing_dept",
    "corporate_strategy_org",
    "consulting_project_sponsor"
]

# Transformation intent dimensions
TRANSFORMATION_INTENT_DIMENSIONS = [
    "strategic_transformation_partner",
    "AI_automation_workflow_product",
    "payer_risk_operator",
    "medical_device",
    "platform_vendor"
]


def economic_mode_vector(extracted_data: Dict) -> np.ndarray:
    """
    Extract economic mode vector from LLM extraction.
    
    Maps revenue_channels and delivery_modes to economic mode dimensions.
    Falls back to inferring from revenue_model if revenue_channels missing.
    
    Args:
        extracted_data: Dict with revenue_channels, delivery_modes, revenue_model
    
    Returns:
        Normalized vector of length len(ECONOMIC_MODE_DIMENSIONS)
    """
    vec = np.zeros(len(ECONOMIC_MODE_DIMENSIONS), dtype=float)
    
    revenue_channels = extracted_data.get('revenue_channels', {})
    delivery_modes = extracted_data.get('delivery_modes', [])
    revenue_model = extracted_data.get('revenue_model', [])
    
    # If revenue_channels is missing, infer from revenue_model
    if not revenue_channels and revenue_model:
        # Infer revenue_channels from revenue_model (backward compatibility)
        revenue_channels = {}
        if 'project_fees' in revenue_model or 'time_and_materials' in revenue_model:
            revenue_channels['professional_services_project'] = 0.7
        if 'managed_services' in revenue_model:
            revenue_channels['managed_services'] = 0.3
        if 'subscription_software' in revenue_model:
            revenue_channels['subscription_recurring'] = 0.5
    
    # hourly_billable: time_and_materials, professional_services_project
    vec[0] = (
        revenue_channels.get('professional_services_project', 0.0) +
        (0.5 if 'time_and_materials' in revenue_model else 0.0)
    )
    
    # fixed_fee_projects: project_fees, professional_services_project (fixed portion)
    vec[1] = (
        revenue_channels.get('professional_services_project', 0.0) * 0.5 +  # Half of PS is fixed
        (1.0 if 'project_fees' in revenue_model else 0.0)
    )
    
    # managed_services: managed_services channel
    vec[2] = revenue_channels.get('managed_services', 0.0)
    
    # subscription_software: subscription_recurring, subscription_software
    vec[3] = (
        revenue_channels.get('subscription_recurring', 0.0) +
        (1.0 if 'subscription_software' in revenue_model else 0.0) +
        (1.0 if 'cloud_saas' in delivery_modes else 0.0) * 0.5
    )
    
    # performance_based: performance_outcome archetype
    vec[4] = extracted_data.get('revenue_archetypes', {}).get('performance_outcome', 0.0)
    
    # risk_bearing: performance_outcome + managed_services (if risk-bearing)
    vec[5] = (
        extracted_data.get('revenue_archetypes', {}).get('performance_outcome', 0.0) * 0.5 +
        (revenue_channels.get('managed_services', 0.0) if 'risk' in str(extracted_data.get('business_activity', [])).lower() else 0.0)
    )
    
    # If still no data, infer from business_model_type
    if vec.sum() == 0:
        business_model_type = extracted_data.get('business_model_type', 'other')
        if business_model_type == 'services':
            vec[0] = 0.4  # hourly_billable
            vec[1] = 0.4  # fixed_fee_projects
            vec[2] = 0.2  # managed_services
        elif business_model_type == 'software':
            vec[3] = 0.9  # subscription_software
            vec[0] = 0.1  # hourly_billable (PS around software)
        elif business_model_type == 'hybrid_services_software':
            services_share = extracted_data.get('services_share_estimate', 0.5)
            vec[0] = 0.3 * services_share  # hourly_billable
            vec[1] = 0.3 * services_share  # fixed_fee_projects
            vec[2] = 0.2 * services_share  # managed_services
            vec[3] = 1.0 - services_share  # subscription_software
    
    # Normalize to sum to 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    else:
        # Default: equal distribution if no data
        vec = np.ones(len(ECONOMIC_MODE_DIMENSIONS), dtype=float) / len(ECONOMIC_MODE_DIMENSIONS)
    
    return vec


def deal_structure_vector(extracted_data: Dict) -> np.ndarray:
    """
    Extract deal structure vector from LLM extraction.
    
    Falls back to inferring from revenue_model if revenue_channels missing.
    
    Args:
        extracted_data: Dict with revenue_channels, delivery_modes, revenue_model
    
    Returns:
        Normalized vector of length len(DEAL_STRUCTURE_DIMENSIONS)
    """
    vec = np.zeros(len(DEAL_STRUCTURE_DIMENSIONS), dtype=float)
    
    revenue_channels = extracted_data.get('revenue_channels', {})
    delivery_modes = extracted_data.get('delivery_modes', [])
    revenue_model = extracted_data.get('revenue_model', [])
    business_activity = extracted_data.get('business_activity', [])
    business_activity_text = ' '.join([str(a).lower() for a in business_activity if a])
    
    # If revenue_channels is missing, infer from revenue_model
    if not revenue_channels and revenue_model:
        revenue_channels = {}
        if 'project_fees' in revenue_model or 'time_and_materials' in revenue_model:
            revenue_channels['professional_services_project'] = 0.7
        if 'managed_services' in revenue_model:
            revenue_channels['managed_services'] = 0.3
        if 'subscription_software' in revenue_model:
            revenue_channels['subscription_recurring'] = 0.5
    
    # MSA_multiyear: managed_services, retainers, multi-year contracts
    vec[0] = (
        revenue_channels.get('managed_services', 0.0) +
        (1.0 if 'retainers' in revenue_model else 0.0) +
        (0.5 if 'multi-year' in business_activity_text or 'msa' in business_activity_text else 0.0)
    )
    
    # project_based_SOW: professional_services_project, project_fees
    vec[1] = (
        revenue_channels.get('professional_services_project', 0.0) +
        (1.0 if 'project_fees' in revenue_model else 0.0) +
        (0.5 if 'sow' in business_activity_text or 'statement of work' in business_activity_text else 0.0)
    )
    
    # platform_license: license_upfront, perpetual_license
    vec[2] = (
        revenue_channels.get('license_upfront', 0.0) +
        (1.0 if 'perpetual_license' in revenue_model else 0.0)
    )
    
    # per_member_per_month: PMPM pricing (healthcare payer models)
    vec[3] = (
        (1.0 if 'pmpm' in business_activity_text or 'per member per month' in business_activity_text else 0.0) +
        (0.5 if 'member' in business_activity_text and 'month' in business_activity_text else 0.0)
    )
    
    # per_provider: per-provider pricing (healthcare)
    vec[4] = (
        (1.0 if 'per provider' in business_activity_text or 'per-provider' in business_activity_text else 0.0) +
        (0.5 if 'provider' in business_activity_text and ('subscription' in business_activity_text or 'license' in business_activity_text) else 0.0)
    )
    
    # per_seat_saas: subscription_recurring, cloud_saas
    vec[5] = (
        revenue_channels.get('subscription_recurring', 0.0) +
        (1.0 if 'cloud_saas' in delivery_modes else 0.0) +
        (0.5 if 'per seat' in business_activity_text or 'per-user' in business_activity_text else 0.0)
    )
    
    # transaction_based: transaction_fees, usage_based
    vec[6] = (
        revenue_channels.get('transaction_fees', 0.0) +
        revenue_channels.get('usage_based', 0.0)
    )
    
    # If still no data, infer from business_model_type
    if vec.sum() == 0:
        business_model_type = extracted_data.get('business_model_type', 'other')
        if business_model_type == 'services':
            vec[0] = 0.3  # MSA_multiyear
            vec[1] = 0.7  # project_based_SOW
        elif business_model_type == 'software':
            vec[2] = 0.2  # platform_license
            vec[5] = 0.8  # per_seat_saas
        elif business_model_type == 'hybrid_services_software':
            services_share = extracted_data.get('services_share_estimate', 0.5)
            vec[0] = 0.2 * services_share  # MSA_multiyear
            vec[1] = 0.5 * services_share  # project_based_SOW
            vec[5] = (1.0 - services_share) * 0.8  # per_seat_saas
    
    # Normalize to sum to 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    else:
        # Default: equal distribution if no data
        vec = np.ones(len(DEAL_STRUCTURE_DIMENSIONS), dtype=float) / len(DEAL_STRUCTURE_DIMENSIONS)
    
    return vec


def buyer_persona_vector(extracted_data: Dict) -> np.ndarray:
    """
    Extract buyer persona vector from LLM extraction.
    
    Infers buyer personas from customer_segment, primary_customer_types, and business_activity.
    
    Args:
        extracted_data: Dict with customer_segment, primary_customer_types, business_activity
    
    Returns:
        Normalized vector of length len(BUYER_PERSONA_DIMENSIONS)
    """
    vec = np.zeros(len(BUYER_PERSONA_DIMENSIONS), dtype=float)
    
    customer_segment = extracted_data.get('customer_segment', [])
    primary_customer_types = extracted_data.get('primary_customer_types', [])
    business_activity = extracted_data.get('business_activity', [])
    
    # Combine all text
    all_text = ' '.join([
        ' '.join([str(c).lower() for c in customer_segment if c]),
        ' '.join([str(c).lower() for c in primary_customer_types if c]),
        ' '.join([str(a).lower() for a in business_activity if a])
    ])
    
    # CIO: enterprise IT, technology transformation
    vec[0] = (
        1.0 if any(kw in all_text for kw in ['cio', 'chief information officer', 'it executive', 'technology leader']) else 0.0
    )
    
    # CFO: financial, revenue cycle, financial advisory
    vec[1] = (
        1.0 if any(kw in all_text for kw in ['cfo', 'chief financial officer', 'financial executive', 'revenue cycle', 'financial advisory']) else 0.0
    )
    
    # provost: higher education, universities
    vec[2] = (
        1.0 if any(kw in all_text for kw in ['provost', 'university', 'higher education', 'academic', 'college']) else 0.0
    )
    
    # hospital_COO: healthcare operations, hospital operations
    vec[3] = (
        1.0 if any(kw in all_text for kw in ['hospital coo', 'hospital operations', 'health system coo', 'healthcare operations']) else 0.0
    )
    
    # network_manager: payer networks, managed care
    vec[4] = (
        1.0 if any(kw in all_text for kw in ['network manager', 'payer network', 'managed care', 'health plan']) else 0.0
    )
    
    # clinic_owner: small practice, clinic owner
    vec[5] = (
        1.0 if any(kw in all_text for kw in ['clinic owner', 'practice owner', 'small practice', 'independent practice']) else 0.0
    )
    
    # IT_system_admin: IT admin, system administrator
    vec[6] = (
        1.0 if any(kw in all_text for kw in ['it admin', 'system administrator', 'it support', 'technical support']) else 0.0
    )
    
    # claims_adjuster: claims processing, payer operations
    vec[7] = (
        1.0 if any(kw in all_text for kw in ['claims adjuster', 'claims processing', 'payer operations', 'claims adjudication']) else 0.0
    )
    
    # EMR_billing_dept: EMR billing, revenue cycle department
    vec[8] = (
        1.0 if any(kw in all_text for kw in ['emr billing', 'billing department', 'revenue cycle department', 'medical billing']) else 0.0
    )
    
    # corporate_strategy_org: corporate strategy, transformation
    vec[9] = (
        1.0 if any(kw in all_text for kw in ['corporate strategy', 'strategy organization', 'transformation', 'organizational change']) else 0.0
    )
    
    # consulting_project_sponsor: consulting, advisory, professional services
    vec[10] = (
        1.0 if any(kw in all_text for kw in ['consulting', 'advisory', 'professional services', 'transformation consulting']) else 0.0
    )
    
    # Normalize to sum to 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    else:
        # Default: equal distribution if no data
        vec = np.ones(len(BUYER_PERSONA_DIMENSIONS), dtype=float) / len(BUYER_PERSONA_DIMENSIONS)
    
    return vec


def transformation_intent_vector(extracted_data: Dict) -> np.ndarray:
    """
    Extract transformation intent vector from LLM extraction.
    
    Classifies company archetype: transformation partner vs automation product vs payer operator, etc.
    
    Args:
        extracted_data: Dict with business_activity, core_consulting_offerings, business_model_type
    
    Returns:
        Normalized vector of length len(TRANSFORMATION_INTENT_DIMENSIONS)
    """
    vec = np.zeros(len(TRANSFORMATION_INTENT_DIMENSIONS), dtype=float)
    
    business_activity = extracted_data.get('business_activity', [])
    core_consulting = extracted_data.get('core_consulting_offerings', [])
    business_model_type = extracted_data.get('business_model_type', 'other')
    
    # Combine all text
    all_text = ' '.join([
        ' '.join([str(a).lower() for a in business_activity if a]),
        ' '.join([str(c).lower() for c in core_consulting if c])
    ])
    
    # strategic_transformation_partner: consulting, advisory, transformation, strategy
    if any(kw in all_text for kw in ['transformation', 'strategic', 'advisory', 'consulting', 'organizational change', 'performance improvement']):
        vec[0] = 1.0
    elif business_model_type == 'services':
        vec[0] = 1.0
    else:
        vec[0] = 0.0
    
    # AI_automation_workflow_product: automation, workflow, AI product, platform
    if any(kw in all_text for kw in ['automation', 'workflow', 'ai product', 'platform', 'saas', 'software product', 'analytics platform', 'data platform']):
        vec[1] = 1.0
    elif business_model_type == 'software':
        vec[1] = 1.0
    else:
        vec[1] = 0.0
    
    # payer_risk_operator: risk-bearing, managed care, payer operations
    vec[2] = (
        1.0 if any(kw in all_text for kw in ['risk-bearing', 'managed care', 'payer operations', 'value-based care', 'utilization management']) else 0.0
    )
    
    # medical_device: medical device, hardware, equipment
    vec[3] = (
        1.0 if any(kw in all_text for kw in ['medical device', 'hardware', 'equipment', 'diagnostic']) else 0.0
    ) or (1.0 if business_model_type == 'hardware' else 0.0)
    
    # platform_vendor: platform, marketplace, infrastructure
    vec[4] = (
        1.0 if any(kw in all_text for kw in ['platform vendor', 'marketplace', 'infrastructure', 'enterprise platform']) else 0.0
    ) or (1.0 if business_model_type == 'marketplace' else 0.0)
    
    # Normalize to sum to 1.0
    total = vec.sum()
    if total > 0:
        vec = vec / total
    else:
        # Default: equal distribution if no data
        vec = np.ones(len(TRANSFORMATION_INTENT_DIMENSIONS), dtype=float) / len(TRANSFORMATION_INTENT_DIMENSIONS)
    
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def compute_economic_similarities(
    candidate_data: Dict,
    target_profile: Dict
) -> Dict[str, float]:
    """
    Compute all economic similarity scores between candidate and target.
    
    Returns:
        Dict with:
            - economic_mode_similarity: float [0, 1]
            - deal_structure_similarity: float [0, 1]
            - buyer_persona_similarity: float [0, 1]
            - transformation_intent_similarity: float [0, 1]
            - overall_economic_similarity: float [0, 1] (weighted average)
    """
    # Compute vectors for candidate and target
    cand_econ = economic_mode_vector(candidate_data)
    target_econ = economic_mode_vector(target_profile)
    
    cand_deal = deal_structure_vector(candidate_data)
    target_deal = deal_structure_vector(target_profile)
    
    cand_buyer = buyer_persona_vector(candidate_data)
    target_buyer = buyer_persona_vector(target_profile)
    
    cand_trans = transformation_intent_vector(candidate_data)
    target_trans = transformation_intent_vector(target_profile)
    
    # Compute similarities
    econ_sim = cosine_similarity(cand_econ, target_econ)
    deal_sim = cosine_similarity(cand_deal, target_deal)
    buyer_sim = cosine_similarity(cand_buyer, target_buyer)
    trans_sim = cosine_similarity(cand_trans, target_trans)
    
    # Overall economic similarity (weighted average)
    # Transformation intent is MOST CRITICAL (kills SaaS vs consulting mismatch)
    # If transformation intent is 0, the company is fundamentally different (SaaS vs consulting)
    # Economic mode is second (kills hourly vs subscription mismatch)
    overall = (
        0.50 * trans_sim +  # MOST CRITICAL: transformation vs automation (raised from 0.35)
        0.25 * econ_sim +   # Second: economic engine (hourly vs subscription)
        0.15 * deal_sim +   # Third: deal structure (MSA vs per-seat)
        0.10 * buyer_sim    # Fourth: buyer persona (provost vs IT admin)
    )
    
    # SPECIAL RULE: If transformation intent similarity is very low (< 0.2), 
    # heavily penalize overall similarity (this kills SaaS vs consulting mismatch)
    if trans_sim < 0.2:
        # Apply strong penalty: multiply overall by transformation similarity
        # If trans_sim = 0.0, overall goes to near 0
        overall = overall * (0.3 + 0.7 * trans_sim)  # If trans_sim=0, overall *= 0.3
    
    return {
        'economic_mode_similarity': econ_sim,
        'deal_structure_similarity': deal_sim,
        'buyer_persona_similarity': buyer_sim,
        'transformation_intent_similarity': trans_sim,
        'overall_economic_similarity': overall
    }

