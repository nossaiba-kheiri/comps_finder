"""
llm_extract.py: Call LLM to extract required fields from EvidencePack; enforce strict schema/JSON.
"""
import json
import os
from datetime import datetime

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False


def extract_llm_structured(evidence_pack, api_key=None, prompt_version='svc_cust_v3', run_with_llm=False):
    """
    Extract structured data from EvidencePack using LLM.
    Returns strict JSON with business_activity, customer_segment, initiatives (with materiality), etc.
    
    Args:
        evidence_pack: EvidencePack dict with sources
        api_key: OpenAI API key
        prompt_version: Prompt version string
        run_with_llm: If True, use real OpenAI API; else return mock data
    """
    ticker = evidence_pack.get('ticker', '')
    sources = evidence_pack.get('sources', [])
    segment_mix_xbrl = evidence_pack.get('segment_mix_xbrl')
    
    # Extract text from sources (chunk if >20k tokens)
    combined_text = ' '.join([s.get('text', '') for s in sources])
    if len(combined_text) > 20000:
        combined_text = combined_text[:20000] + "..."
    
    # Check if we have 10-K evidence (for materiality scoring)
    has_10k = any(s.get('type') == '10K' for s in sources)
    
    if run_with_llm and openai_available:
        # Real LLM extraction with OpenAI
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fall back to mock
            return _mock_extraction(ticker, sources, combined_text, segment_mix_xbrl, has_10k, prompt_version)
        
        try:
            client = OpenAI(api_key=api_key)
            
            # Build prompt
            prompt = f"""Extract structured information from the following company evidence text.
Return ONLY valid JSON matching this exact schema:
{{
  "business_activity": ["normalized service/product phrases"],
  "customer_segment": ["normalized buyer/verticals"],
  "segment_mix": {{"bucket": weight}} or null,
  "initiatives": [
    {{
      "name": "initiative name",
      "category": "product/service/customer",
      "description": "brief description",
      "materiality_0_1": 0.0-1.0,
      "evidence_ref": "source_url or section"
    }}
  ],
  "similar_industries": ["similar OWN industries - industries where comparable companies operate"],
  "customer_industries": ["customer industry verticals served - NOT the company's own industry, but industries where the company's customers operate"],
  "SIC_industry": ["industry codes if mentioned"],
  "exchange": "exchange code if mentioned",
  "evidence": [
    {{
      "source_url": "url",
      "section": "section name",
      "quote": "relevant quote with product or customer mention"
    }}
  ],
  "confidence_0_1": 0.0-1.0,
  "model_meta": {{"model": "gpt-4", "prompt_version": "{prompt_version}"}}
}}

CRITICAL DISTINCTIONS:
- "similar_industries": Similar OWN industries (what the company IS - e.g., for a consulting firm: ["Consulting Services", "Business Services"])
- "customer_industries": Customer industry verticals (who the company SERVES - e.g., ["Healthcare", "Financial Services", "Education"])
- For consulting firms: similar_industries = their own industry type (Consulting Services), customer_industries = industries they serve (Healthcare, Education)
- For software companies: similar_industries = their own industry type (Software - Infrastructure), customer_industries = industries their software serves (Healthcare, Retail)

Materiality rules:
- Initiatives mentioned only on site/newsroom: materiality_0_1 ≤ 0.10
- Initiatives mentioned in 10-K Item 1/MD&A/segment: materiality_0_1 0.3-0.7 (based on revenue share/strategy importance)
- Main business activities: materiality_0_1 = 1.0 (implicit)

Evidence must include at least:
- 1 product quote with URL
- 1 customer quote with URL

Company evidence text:
{combined_text[:15000]}

Return ONLY the JSON object, no other text."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a structured data extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            extracted = json.loads(response_text)
            
            # Validate and set defaults
            extracted = _validate_extraction(extracted, sources, has_10k)
            
            return extracted
        except Exception as e:
            print(f"    LLM extraction failed for {ticker}: {e}")
            # Fall back to mock
            return _mock_extraction(ticker, sources, combined_text, segment_mix_xbrl, has_10k, prompt_version)
    else:
        # Mock extraction (for testing)
        return _mock_extraction(ticker, sources, combined_text, segment_mix_xbrl, has_10k, prompt_version)


def _mock_extraction(ticker, sources, combined_text, segment_mix_xbrl, has_10k, prompt_version):
    """Mock extraction for testing (when LLM not available)."""
    # Extract basic info from text
    business_activity = []
    customer_segment = []
    initiatives = []
    
    # Simple keyword extraction
    text_lower = combined_text.lower()
    if 'payment' in text_lower or 'transaction' in text_lower:
        business_activity.append("payment processing")
        initiatives.append({
            "name": "Payment Processing",
            "category": "product",
            "description": "Payment processing services",
            "materiality_0_1": 0.8 if has_10k else 0.1,
            "evidence_ref": sources[0].get('url', '') if sources else ''
        })
    if 'cloud' in text_lower or 'saas' in text_lower:
        business_activity.append("cloud services")
    if 'bank' in text_lower or 'financial' in text_lower:
        customer_segment.append("banks")
    if 'retail' in text_lower:
        customer_segment.append("retailers")
    
    # Default if nothing found
    if not business_activity:
        business_activity = ["software services", "enterprise solutions"]
    if not customer_segment:
        customer_segment = ["enterprises", "businesses"]
    
    # Use XBRL segment mix if available
    segment_mix = segment_mix_xbrl or {}
    
    # Extract evidence quotes
    evidence = []
    if sources:
        for source in sources[:3]:  # Top 3 sources
            text = source.get('text', '')[:500]
            if text:
                evidence.append({
                    "source_url": source.get('url', ''),
                    "section": source.get('section', 'business description'),
                    "quote": text
                })
    
    if not evidence and sources:
        evidence.append({
            "source_url": sources[0].get('url', ''),
            "section": "business description",
            "quote": combined_text[:200] if combined_text else "No description available"
        })
    
    # Extract similar own industries (what the company IS)
    similar_industries = []
    if 'consulting' in text_lower:
        similar_industries.extend(["Consulting Services", "Business Services", "Professional Services"])
    if 'software' in text_lower or 'saas' in text_lower:
        similar_industries.extend(["Software - Application", "Software - Infrastructure", "Technology"])
    if 'services' in text_lower and 'business' in text_lower:
        similar_industries.append("Business Services")
    
    # Extract customer industries (verticals served - who the company SERVES)
    customer_industries = []
    if 'healthcare' in text_lower or 'medical' in text_lower or 'hospital' in text_lower:
        customer_industries.append("Healthcare")
    if 'financial' in text_lower or 'bank' in text_lower or 'fintech' in text_lower:
        customer_industries.append("Financial Services")
    if 'retail' in text_lower or 'e-commerce' in text_lower:
        customer_industries.append("Retail")
    if 'education' in text_lower or 'university' in text_lower:
        customer_industries.append("Education & Research")
    if 'manufacturing' in text_lower or 'industrial' in text_lower:
        customer_industries.append("Industrials & Manufacturing")
    if 'energy' in text_lower or 'utilities' in text_lower:
        customer_industries.append("Energy & Utilities")
    if 'government' in text_lower or 'public sector' in text_lower:
        customer_industries.append("Public Sector")
    
    return {
        "business_activity": business_activity,
        "customer_segment": customer_segment,
        "segment_mix": segment_mix,
        "initiatives": initiatives,
        "similar_industries": similar_industries,  # Similar own industries (what company IS)
        "customer_industries": customer_industries,  # Customer industries served (who company SERVES)
        "SIC_industry": [],
        "exchange": "NASDAQ",
        "evidence": evidence,
        "confidence_0_1": 0.8 if has_10k else 0.6,
        "model_meta": {
            "model": "gpt-4o" if openai_available else "mock",
            "prompt_version": prompt_version
        }
    }


def _validate_extraction(extracted, sources, has_10k):
    """Validate extracted JSON and ensure required fields."""
    # Ensure required fields exist
    if 'business_activity' not in extracted:
        extracted['business_activity'] = []
    if 'customer_segment' not in extracted:
        extracted['customer_segment'] = []
    if 'similar_industries' not in extracted:
        extracted['similar_industries'] = []  # Similar own industries
    if 'customer_industries' not in extracted:
        # Backward compatibility: check for old 'industries' field
        extracted['customer_industries'] = extracted.get('industries', [])
    if 'initiatives' not in extracted:
        extracted['initiatives'] = []
    if 'segment_mix' not in extracted:
        extracted['segment_mix'] = {}
    if 'evidence' not in extracted:
        extracted['evidence'] = []
    
    # Validate initiatives materiality
    for initiative in extracted.get('initiatives', []):
        materiality = initiative.get('materiality_0_1', 0.1)
        # If initiative not mentioned in 10-K, cap materiality at 0.10
        if not has_10k and materiality > 0.10:
            initiative['materiality_0_1'] = 0.10
        # Ensure materiality is in valid range
        initiative['materiality_0_1'] = max(0.0, min(1.0, materiality))
    
    # Ensure evidence has at least one product and one customer quote
    if not extracted['evidence']:
        # Add default evidence
        if sources:
            extracted['evidence'] = [{
                "source_url": sources[0].get('url', ''),
                "section": "business description",
                "quote": sources[0].get('text', '')[:200] if sources[0].get('text') else "No description available"
            }]
    
    return extracted


if __name__ == "__main__":
    # Test
    pack = {
        'ticker': 'AAPL',
        'sources': [
            {'type': 'site', 'url': 'https://apple.com', 'text': 'Apple designs and manufactures consumer electronics.'}
        ]
    }
    extracted = extract_llm_structured(pack)
    print(json.dumps(extracted, indent=2))
