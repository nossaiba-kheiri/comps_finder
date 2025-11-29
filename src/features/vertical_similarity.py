"""
vertical_similarity.py: Vertical similarity scoring (V), using multi-hot encoding of vertical categories.

This feature encodes companies as multi-hot vectors of vertical/industry categories they serve:
- healthcare
- education
- government
- defense
- financials
- manufacturing
- retail
- etc.

Then computes cosine similarity between target and candidate vertical vectors.

This is more accurate than matching generic terms like "consulting" - it focuses on
the specific verticals/industries companies actually serve.
"""
import numpy as np
from typing import Dict, List, Set, Optional
import re

# Define vertical taxonomy (multi-hot categories)
VERTICAL_CATEGORIES = [
    'healthcare',
    'education',
    'government',
    'defense',
    'financials',
    'manufacturing',
    'retail',
    'energy',
    'telecommunications',
    'technology',
    'media',
    'real_estate',
    'transportation',
    'hospitality',
    'agriculture',
    'pharmaceuticals',
    'life_sciences',
    'aerospace',
    'automotive',
    'construction',
    'utilities',
    'consumer_goods',
    'professional_services',
    'nonprofit',
    'legal',
    'insurance',
]

# Keywords/phrases that indicate vertical exposure
VERTICAL_KEYWORDS = {
    'healthcare': [
        'healthcare', 'health care', 'hospital', 'medical', 'clinical', 'patient care',
        'health system', 'healthcare system', 'provider', 'physician', 'nurse',
        'healthcare delivery', 'medical center', 'healthcare organization',
        'revenue cycle management', 'rcm', 'clinical workflow', 'healthcare analytics',
        'healthcare operations', 'digital health', 'healthcare technology'
    ],
    'education': [
        'education', 'university', 'college', 'school', 'higher education',
        'k-12', 'k12', 'student', 'academic', 'campus', 'university system',
        'educational institution', 'learning', 'education technology', 'edtech',
        'student information system', 'enrollment', 'university enterprise systems'
    ],
    'government': [
        'government', 'gov', 'public sector', 'federal', 'state', 'local government',
        'municipal', 'public administration', 'government agency', 'public service',
        'gov enrollment services', 'government services', 'public sector consulting'
    ],
    'defense': [
        'defense', 'military', 'defense contractor', 'department of defense', 'dod',
        'armed forces', 'national security', 'defense industry'
    ],
    'financials': [
        'financial', 'finance', 'banking', 'bank', 'financial services',
        'investment', 'wealth management', 'asset management', 'capital markets',
        'insurance', 'fintech', 'financial institution'
    ],
    'manufacturing': [
        'manufacturing', 'industrial', 'factory', 'production', 'supply chain',
        'manufacturing operations', 'industrial manufacturing'
    ],
    'retail': [
        'retail', 'retailer', 'retail operations', 'consumer retail', 'e-commerce',
        'omnichannel', 'retail technology'
    ],
    'energy': [
        'energy', 'utilities', 'power', 'electric', 'oil', 'gas', 'renewable energy',
        'energy sector', 'utility company'
    ],
    'telecommunications': [
        'telecommunications', 'telecom', 'communications', 'wireless', 'network',
        'telecom provider', 'communications provider'
    ],
    'technology': [
        'technology', 'tech', 'software', 'it', 'information technology',
        'technology services', 'tech company'
    ],
    'media': [
        'media', 'entertainment', 'broadcasting', 'publishing', 'content',
        'media company', 'entertainment industry'
    ],
    'real_estate': [
        'real estate', 'realty', 'property', 'commercial real estate',
        'residential real estate', 'property management'
    ],
    'transportation': [
        'transportation', 'logistics', 'shipping', 'freight', 'supply chain',
        'transportation services', 'logistics provider'
    ],
    'hospitality': [
        'hospitality', 'hotel', 'restaurant', 'travel', 'tourism', 'hospitality industry'
    ],
    'agriculture': [
        'agriculture', 'agricultural', 'farming', 'agribusiness', 'crop', 'livestock'
    ],
    'pharmaceuticals': [
        'pharmaceutical', 'pharma', 'drug', 'biotech', 'pharmaceutical company'
    ],
    'life_sciences': [
        'life sciences', 'biotechnology', 'biotech', 'life science', 'biomedical'
    ],
    'aerospace': [
        'aerospace', 'aviation', 'aircraft', 'aerospace industry'
    ],
    'automotive': [
        'automotive', 'automobile', 'vehicle', 'auto', 'car manufacturing'
    ],
    'construction': [
        'construction', 'construction industry', 'building', 'infrastructure'
    ],
    'utilities': [
        'utilities', 'utility', 'electric utility', 'water utility', 'gas utility'
    ],
    'consumer_goods': [
        'consumer goods', 'consumer products', 'cp', 'consumer packaged goods'
    ],
    'professional_services': [
        'professional services', 'consulting', 'advisory', 'professional services firm'
    ],
    'nonprofit': [
        'nonprofit', 'non-profit', 'non profit', 'charity', 'foundation', 'ngo'
    ],
    'legal': [
        'legal', 'law', 'law firm', 'legal services', 'attorney', 'lawyer'
    ],
    'insurance': [
        'insurance', 'insurer', 'insurance company', 'insurance provider'
    ],
}


def extract_vertical_exposure(
    company_text: str,
    customer_segments: Optional[List[str]] = None,
    customer_industries: Optional[List[str]] = None,
    business_activity: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extract vertical exposure from company metadata.
    
    PRIMARY FOCUS: customer_segments and customer_industries (verticals served)
    SECONDARY: business_activity (can indicate vertical focus)
    TERTIARY: company_text (fallback if structured data missing)
    
    Returns multi-hot vector where each element is 1 if company serves that vertical, 0 otherwise.
    
    Args:
        company_text: Company description/summary text (used as fallback)
        customer_segments: List of customer segment strings (PRIMARY - verticals served)
        customer_industries: List of customer industry strings (PRIMARY - verticals served)
        business_activity: List of business activity strings (SECONDARY - can indicate vertical focus)
    
    Returns:
        numpy array of shape (len(VERTICAL_CATEGORIES),) with values 0 or 1
    """
    # PRIORITY 1: Use customer_segments and customer_industries (explicit verticals served)
    # This is the PRIMARY signal - directly tells us which verticals the company serves
    # Example: customer_segment=["healthcare systems", "universities"] → serves healthcare, education
    primary_text = []
    if customer_segments:
        primary_text.extend([str(s).lower() for s in customer_segments])
    if customer_industries:
        primary_text.extend([str(i).lower() for i in customer_industries])
    
    # PRIORITY 2: Use business_activity ONLY if primary data is sparse
    # Business activity can indicate vertical focus (e.g., "healthcare digital ops" → healthcare)
    # But it's indirect and less reliable than explicit customer_segment
    # Only use if we don't have enough primary data
    secondary_text = []
    if business_activity and len(primary_text) < 2:
        # Only use business_activity if we have < 2 customer segments/industries
        # This ensures we prioritize explicit vertical declarations
        secondary_text.extend([str(a).lower() for a in business_activity])
    
    # PRIORITY 3: Use company_text as fallback (only if structured data is very sparse)
    fallback_text = []
    if company_text and len(primary_text) == 0 and len(secondary_text) == 0:
        # Only use company_text if we have NO structured data at all
        fallback_text.append(str(company_text).lower())
    
    # Combine: primary first, then secondary (only if needed), then fallback (only if needed)
    combined_text = ' '.join(primary_text + secondary_text + fallback_text)
    
    # Initialize multi-hot vector
    vertical_vector = np.zeros(len(VERTICAL_CATEGORIES), dtype=np.float32)
    
    # Check each vertical category
    for idx, vertical in enumerate(VERTICAL_CATEGORIES):
        keywords = VERTICAL_KEYWORDS.get(vertical, [])
        
        # Check if any keyword appears in the text
        for keyword in keywords:
            # Use word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, combined_text, re.IGNORECASE):
                vertical_vector[idx] = 1.0
                break  # Found a match, no need to check other keywords for this vertical
    
    return vertical_vector


def compute_vertical_similarity(
    target_vector: np.ndarray,
    candidate_vector: np.ndarray
) -> float:
    """
    Compute cosine similarity between two vertical exposure vectors.
    
    Args:
        target_vector: Multi-hot vector of target's vertical exposure
        candidate_vector: Multi-hot vector of candidate's vertical exposure
    
    Returns:
        Cosine similarity score in [0, 1]
    """
    # Handle edge cases
    if np.sum(target_vector) == 0 or np.sum(candidate_vector) == 0:
        # If either vector is all zeros, return 0 (no vertical overlap)
        return 0.0
    
    # Compute cosine similarity
    dot_product = np.dot(target_vector, candidate_vector)
    norm_target = np.linalg.norm(target_vector)
    norm_candidate = np.linalg.norm(candidate_vector)
    
    if norm_target == 0 or norm_candidate == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm_target * norm_candidate)
    
    # Clamp to [0, 1]
    return float(np.clip(cosine_sim, 0.0, 1.0))


def score_vertical_similarity(
    target_data: Dict,
    candidate_data: Dict,
    candidate_text: Optional[str] = None
) -> float:
    """
    Compute vertical similarity score V between target and candidate.
    
    Args:
        target_data: Target company dict with customer_segment, customer_industries, business_activity, etc.
        candidate_data: Candidate company dict with same fields
        candidate_text: Optional candidate description text (if not in candidate_data)
    
    Returns:
        Vertical similarity score in [0, 1]
    """
    # Extract target vertical exposure
    # PRIMARY: Use customer_segment and customer_industries (verticals served)
    # This is the key - we want to match companies that serve the same verticals
    target_text = target_data.get('raw_profile_text', '') or target_data.get('summary', '')
    target_vector = extract_vertical_exposure(
        company_text=target_text,
        customer_segments=target_data.get('customer_segment', []),  # PRIMARY: verticals served
        customer_industries=target_data.get('customer_industries', []),  # PRIMARY: verticals served
        business_activity=target_data.get('business_activity', [])  # SECONDARY: can indicate vertical focus
    )
    
    # Extract candidate vertical exposure
    # PRIMARY: Use customer_segment and customer_industries (verticals served)
    candidate_text_source = candidate_text or candidate_data.get('summary', '') or candidate_data.get('raw_profile_text', '')
    candidate_vector = extract_vertical_exposure(
        company_text=candidate_text_source,
        customer_segments=candidate_data.get('customer_segment', []),  # PRIMARY: verticals served
        customer_industries=candidate_data.get('customer_industries', []),  # PRIMARY: verticals served
        business_activity=candidate_data.get('business_activity', [])  # SECONDARY: can indicate vertical focus
    )
    
    # Compute cosine similarity
    similarity = compute_vertical_similarity(target_vector, candidate_vector)
    
    return similarity


if __name__ == "__main__":
    # Test
    target = {
        'customer_segment': ['healthcare systems', 'universities', 'government agencies'],
        'customer_industries': ['Healthcare', 'Education', 'Government'],
        'business_activity': ['revenue cycle management', 'healthcare digital ops', 'university enterprise systems'],
        'raw_profile_text': 'Huron serves healthcare, education, and government clients with revenue cycle management and digital operations.'
    }
    
    candidate = {
        'customer_segment': ['healthcare providers', 'hospitals'],
        'customer_industries': ['Healthcare'],
        'business_activity': ['healthcare analytics', 'clinical workflow'],
        'summary': 'Company provides healthcare analytics and clinical workflow solutions to hospitals.'
    }
    
    score = score_vertical_similarity(target, candidate)
    print(f"Vertical similarity: {score:.3f}")
    
    # Test with different verticals
    candidate2 = {
        'customer_segment': ['banks', 'financial institutions'],
        'customer_industries': ['Financials'],
        'business_activity': ['banking software'],
        'summary': 'Company provides banking software to financial institutions.'
    }
    
    score2 = score_vertical_similarity(target, candidate2)
    print(f"Vertical similarity (different verticals): {score2:.3f}")

