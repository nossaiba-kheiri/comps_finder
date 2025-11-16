"""
struct_target.py: Build structured target.json from website + LinkedIn using LLM.

This module (per spec):
1. Takes assignment inputs: name, url, business_description, primary_industry_classification
2. Crawls target website using Apify (filters relevant pages)
3. Fetches last 8 months of LinkedIn posts (filters relevant posts)
4. Builds comprehensive profile_text from all sources
5. Uses LLM to extract structured data matching exact schema
6. No yfinance - only website + LinkedIn for target

Schema:
- name, url, primary_industry_classification, business_description
- business_activity[], customer_segment[], product_mix{}, industries[]
- evidence{}, raw_profile_text
"""
import os
import sys
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from evidence.fetch_apify_website import fetch_website_content
from evidence.fetch_linkedin import fetch_linkedin_posts, extract_company_name_from_url
from nlp.llm_extract import extract_llm_structured

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    openai_available = True
except ImportError:
    openai_available = False


def filter_relevant_pages(pages: List[Dict]) -> List[Dict]:
    """
    Filter website pages to keep only relevant ones.
    
    Keeps pages that describe services, solutions, industries, clients, etc.
    Drops short/boilerplate pages (< 300 chars).
    """
    keep = []
    keywords = ["service", "solution", "industry", "sector", "clients", "what we do", 
                "offering", "capability", "products", "about", "expertise"]
    
    for p in pages:
        text = (p.get("text") or "").lower()
        if len(text) < 300:
            continue  # Skip short/boilerplate pages
        if any(k in text for k in keywords):
            keep.append(p)
    
    return keep


def filter_relevant_linkedin_posts(posts: List[Dict], months_back: int = 8) -> List[Dict]:
    """
    Filter LinkedIn posts to last N months and relevant content.
    
    Filters by:
    - Date (last N months)
    - Content keywords (launches, clients, solutions, etc.)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
    content_keywords = [
        "launched", "launch", "solution", "clients", "partner", "partnership",
        "hospital", "university", "bank", "insurer", "retail", "energy", 
        "portfolio", "service", "offering", "customer", "announce"
    ]
    
    filtered = []
    for p in posts:
        # Filter by date
        posted_at = p.get("postedAt")
        if not posted_at:
            continue
        
        try:
            if isinstance(posted_at, str):
                dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            else:
                continue
        except Exception:
            continue
        
        if dt < cutoff:
            continue
        
        # Filter by content relevance
        content = (p.get("content") or "").lower()
        if any(k in content for k in content_keywords):
            filtered.append(p)
    
    return filtered


def build_target_profile_text(
    name: str,
    business_description: str,
    primary_industry_classification: str,
    pages: List[Dict],
    posts: List[Dict]
) -> str:
    """
    Build comprehensive profile_text from assignment inputs + website + LinkedIn.
    
    Concatenates:
    1. Assignment-level description (name, industry, business_description)
    2. Top N relevant website pages
    3. Top N relevant LinkedIn posts
    """
    parts = []
    
    # 1) Assignment-level description
    parts.append(f"{name} is a company in the '{primary_industry_classification}' industry. {business_description}")
    
    # 2) Website – take top N pages (5-10)
    for page in pages[:10]:
        text = page.get("text", "")
        if text:
            parts.append(text)
    
    # 3) LinkedIn – append posts talking about offerings/clients
    for post in posts[:20]:
        content = post.get("content", "")
        if content:
            parts.append(content)
    
    return "\n\n".join(parts)


def extract_target_structured_llm(
    name: str,
    url: str,
    business_description: str,
    primary_industry_classification: str,
    profile_text: str,
    api_key: str = None
) -> Dict:
    """
    Call OpenAI LLM to extract structured target data with exact schema.
    
    Returns:
        Dict with: name, url, primary_industry_classification, business_description,
                   business_activity[], customer_segment[], product_mix{}, 
                   industries[], evidence{}, raw_profile_text
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are a financial analyst. You read company descriptions, websites and LinkedIn posts and output a structured JSON describing what the business does, who it serves, and how its activity is split across segments.

Assignment Information:
- name: {name}
- url: {url}
- primary_industry_classification: {primary_industry_classification}
- business_description: {business_description}

Profile Text (from website + LinkedIn):
{profile_text[:15000]}

Using ONLY the information in the profile text and assignment fields, extract the following JSON fields:

{{
  "name": "{name}",
  "url": "{url}",
  "primary_industry_classification": "{primary_industry_classification}",
  "business_description": "{business_description}",
  "business_activity": ["normalized product/service phrases"],
  "customer_segment": ["normalized buyer/verticals"],
  "product_mix": {{"segment_name": weight_between_0_and_1}},
  "industries": ["operational industry names"],
  "evidence": {{
    "business_activity": ["quotes from profile_text about products/services"],
    "customer_segment": ["quotes from profile_text about customers"],
    "product_mix": ["quotes from profile_text about segment distribution"]
  }},
  "raw_profile_text": "{profile_text[:5000]}"
}}

Rules:
- Extract 3-7 most important business_activity items
- Extract 2-5 main customer_segment types
- product_mix weights should roughly sum to 1.0 (e.g., {{"healthcare": 0.5, "education": 0.3, "commercial": 0.2}})
- industries should be operational/segmental (not SIC codes)
- evidence should contain actual quotes from profile_text
- Do not invent segments that are not clearly suggested by the text
- Use clear, normalized names

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
    
    response_text = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    extracted = json.loads(response_text)
    return extracted


def build_target_from_website_linkedin(
    name: str,
    url: str,
    business_description: str,
    primary_industry_classification: str,
    linkedin_url: str = None,
    linkedin_company_name: str = None,
    months_back: int = 8,
    api_key: str = None,
    run_with_llm: bool = True
) -> Dict:
    """
    Build structured target.json from assignment inputs + website + LinkedIn.
    
    Per spec:
    - No yfinance for target (only for comps)
    - Uses assignment inputs: name, url, business_description, primary_industry_classification
    - Crawls website (filters relevant pages)
    - Fetches last N months of LinkedIn posts (filters relevant posts)
    - Builds comprehensive profile_text
    - Uses LLM to extract structured data with exact schema
    
    Args:
        name: Company name (from assignment)
        url: Company website URL (from assignment)
        business_description: Brief business description (from assignment)
        primary_industry_classification: Primary industry (from assignment)
        linkedin_url: LinkedIn company URL (optional)
        linkedin_company_name: LinkedIn company vanity name (optional)
        months_back: Number of months to fetch LinkedIn posts (default: 8)
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var)
        run_with_llm: Whether to use real LLM (default: True)
    
    Returns:
        Structured target dict matching exact schema:
        - name, url, primary_industry_classification, business_description
        - business_activity[], customer_segment[], product_mix{}, industries[]
        - evidence{}, raw_profile_text
        - target_id, metadata{}
    """
    print("="*80)
    print("Building Structured Target from Website + LinkedIn")
    print("="*80)
    print(f"Target: {name}")
    print(f"URL: {url}")
    print(f"Industry: {primary_industry_classification}")
    print()
    
    # Generate unique identifier
    target_id = str(uuid.uuid4())
    
    # Step 1: Crawl website
    print(f"[1/3] Crawling website: {url}")
    print(f"  Using Apify with memory-optimized settings...")
    website_pages_raw = fetch_website_content(url, max_pages=10)
    
    # Filter relevant pages
    website_pages = filter_relevant_pages(website_pages_raw)
    print(f"  ✓ Collected {len(website_pages_raw)} pages, filtered to {len(website_pages)} relevant pages")
    
    # Step 2: Fetch LinkedIn posts (last N months)
    print(f"\n[2/3] Fetching LinkedIn posts (last {months_back} months)...")
    posts = []
    try:
        if linkedin_company_name:
            posts_raw = fetch_linkedin_posts(
                company_name=linkedin_company_name,
                months_back=months_back
            )
        elif linkedin_url:
            posts_raw = fetch_linkedin_posts(
                company_url=linkedin_url,
                months_back=months_back
            )
        else:
            print("  No LinkedIn URL/name provided, skipping")
            posts_raw = []
        
        # Filter relevant posts (by date and content)
        posts = filter_relevant_linkedin_posts(posts_raw, months_back=months_back)
        print(f"  ✓ Collected {len(posts_raw)} posts, filtered to {len(posts)} relevant posts")
    except Exception as e:
        error_msg = str(e)
        if "memory limit" in error_msg.lower():
            print(f"  ⚠ Apify memory limit reached - skipping LinkedIn")
            print(f"    (Website data should be sufficient)")
        else:
            print(f"  ⚠ LinkedIn fetch failed: {error_msg[:100]}")
        print("  Continuing without LinkedIn data...")
    
    # Step 3: Build profile_text from all sources
    print(f"\n[3/3] Building profile text and extracting structured data with LLM...")
    profile_text = build_target_profile_text(
        name=name,
        business_description=business_description,
        primary_industry_classification=primary_industry_classification,
        pages=website_pages,
        posts=posts
    )
    print(f"  ✓ Profile text length: {len(profile_text)} characters")
    
    # Step 4: Extract structured data using LLM
    if run_with_llm:
        extracted = extract_target_structured_llm(
            name=name,
            url=url,
            business_description=business_description,
            primary_industry_classification=primary_industry_classification,
            profile_text=profile_text,
            api_key=api_key
        )
    else:
        # Mock extraction for testing
        extracted = {
            "name": name,
            "url": url,
            "primary_industry_classification": primary_industry_classification,
            "business_description": business_description,
            "business_activity": ["Mock Product 1", "Mock Product 2"],
            "customer_segment": ["Mock Customer 1"],
            "product_mix": {"segment1": 1.0},
            "industries": [primary_industry_classification],
            "evidence": {
                "business_activity": [],
                "customer_segment": [],
                "product_mix": []
            },
            "raw_profile_text": profile_text[:5000]
        }
    
    # Step 5: Build final target.json with unique identifier and metadata
    target = {
        # Assignment fields
        "name": extracted.get("name", name),
        "url": extracted.get("url", url),
        "primary_industry_classification": extracted.get("primary_industry_classification", primary_industry_classification),
        "business_description": extracted.get("business_description", business_description),
        
        # Extracted structured data
        "business_activity": extracted.get("business_activity", []),
        "customer_segment": extracted.get("customer_segment", []),
        "product_mix": extracted.get("product_mix", {}),
        "industries": extracted.get("industries", []),
        "evidence": extracted.get("evidence", {}),
        "raw_profile_text": extracted.get("raw_profile_text", profile_text),
        
        # Unique identifier and metadata
        "target_id": target_id,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "created_by": "struct_target.py",
            "version": "1.0",
            "extraction_method": "llm_openai_gpt4o" if run_with_llm else "mock",
            "source": "website_linkedin",
            "source_url": url,
            "primary_industry_classification": primary_industry_classification,
            "months_back_linkedin": months_back,
            "website_pages_count": len(website_pages),
            "linkedin_posts_count": len(posts)
        }
    }
    
    print(f"\n✓ Extracted structured data:")
    print(f"    Business activities: {len(target['business_activity'])}")
    print(f"    Customer segments: {len(target['customer_segment'])}")
    print(f"    Product mix segments: {len(target['product_mix'])}")
    print(f"    Industries: {len(target['industries'])}")
    print(f"    Target ID: {target_id}")
    
    return target


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build structured target.json from website + LinkedIn')
    parser.add_argument('--website', type=str, required=True, help='Company website URL')
    parser.add_argument('--linkedin', type=str, default=None, help='LinkedIn company URL or vanity name')
    parser.add_argument('--ticker', type=str, default=None, help='Company ticker (for 10-K)')
    parser.add_argument('--cik', type=str, default=None, help='Company CIK (for 10-K)')
    parser.add_argument('--use-10k', action='store_true', help='Fetch 10-K if ticker available')
    parser.add_argument('--openai', action='store_true', help='Use real OpenAI LLM')
    parser.add_argument('--output', type=str, default='comps/data/target.json', help='Output file path')
    parser.add_argument('--months', type=int, default=8, help='Months of LinkedIn posts to fetch')
    
    args = parser.parse_args()
    
    # Extract LinkedIn company name if URL provided
    linkedin_company_name = None
    if args.linkedin:
        if "linkedin.com" in args.linkedin:
            linkedin_company_name = extract_company_name_from_url(args.linkedin)
        else:
            linkedin_company_name = args.linkedin
    
    target = build_target_from_website_linkedin(
        website_url=args.website,
        linkedin_url=args.linkedin if "linkedin.com" in (args.linkedin or "") else None,
        linkedin_company_name=linkedin_company_name,
        ticker=args.ticker,
        cik=args.cik,
        months_back=args.months,
        use_10k=args.use_10k,
        run_with_llm=args.openai
    )
    
    # Save to file
    output_path = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(target, f, indent=2)
    
    print(f"\n✓ Saved structured target to {output_path}")

