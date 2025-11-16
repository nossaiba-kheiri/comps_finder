#!/usr/bin/env python3
"""
create_target_from_info.py: Create target.json from assignment inputs + website + LinkedIn.

Inputs (from assignment):
- name
- url (homepage)
- business_description (brief description)
- primary_industry_classification (e.g., "Research and Consulting Services")

Process:
1. Crawl website (Apify website-content-crawler)
2. Scrape LinkedIn posts (last 8 months)
3. Build profile_text from all sources
4. LLM extraction with exact schema
5. Save target.json with unique identifier and metadata

NO yfinance for target - only website + LinkedIn.
"""
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import evidence fetching functions
from evidence.fetch_apify_website import fetch_website_content
from evidence.fetch_linkedin import fetch_linkedin_posts

try:
    from openai import OpenAI
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
    - Date (last N months) - only if not already filtered by fetch_linkedin_posts
    - Content keywords (launches, clients, solutions, etc.)
    """
    # Note: Date filtering is already done in fetch_linkedin_posts
    # This function primarily filters by content relevance
    # But we check date again in case posts were passed without date filtering
    cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)
    content_keywords = [
        "launched", "launch", "solution", "clients", "partner", "partnership",
        "hospital", "university", "bank", "insurer", "retail", "energy", 
        "portfolio", "service", "offering", "customer", "announce",
        # Add more keywords for consulting companies
        "consulting", "advisory", "transformation", "digital", "technology",
        "healthcare", "education", "commercial", "segment", "strategy"
    ]
    
    filtered = []
    date_filtered = 0
    no_keywords = 0
    
    for p in posts:
        # Filter by date (double-check)
        posted_at = p.get("postedAt")
        if not posted_at:
            continue
        
        try:
            if isinstance(posted_at, str):
                # Handle different date formats
                # Format 1: ISO with Z (e.g., "2025-11-14T15:55:09Z")
                if "Z" in posted_at:
                    dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
                # Format 2: Apify format with space (e.g., "2025-11-14 15:55:09")
                elif " " in posted_at and len(posted_at) > 10:
                    dt = datetime.strptime(posted_at, "%Y-%m-%d %H:%M:%S")
                    dt = dt.replace(tzinfo=timezone.utc)  # Make it timezone-aware
                # Format 3: ISO with timezone offset
                elif "+" in posted_at or (posted_at.count("-") > 2 and "T" in posted_at):
                    dt = datetime.fromisoformat(posted_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                # Format 4: ISO without timezone (assume UTC)
                elif "T" in posted_at:
                    dt = datetime.fromisoformat(posted_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                else:
                    # Try ISO format as fallback
                    dt = datetime.fromisoformat(posted_at)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
            else:
                date_filtered += 1
                continue
        except Exception as e:
            date_filtered += 1
            continue
        
        # Ensure dt is timezone-aware for comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        if dt < cutoff:
            date_filtered += 1
            continue
        
        # Filter by content relevance
        content = (p.get("content") or "").lower()
        if any(k in content for k in content_keywords):
            filtered.append(p)
        else:
            no_keywords += 1
    
    # Debug output if no posts passed
    if len(filtered) == 0 and len(posts) > 0:
        print(f"    Debug: {date_filtered} posts filtered by date")
        print(f"    Debug: {no_keywords} posts filtered by content keywords")
        print(f"    Debug: Consider expanding keyword list or reducing keyword filtering")
    
    return filtered


def build_target_profile_text(
    name: str,
    business_description: str,
    primary_industry_classification: str,
    pages: List[Dict],
    posts: List[Dict],
    url: str = None
) -> tuple:
    """
    Build comprehensive profile_text from assignment inputs + website + LinkedIn.
    
    Also returns source mapping to help LLM identify where each part came from.
    
    Returns:
        tuple: (profile_text: str, source_info: dict)
            source_info = {
                "sections": [
                    {"type": "business_description", "start": 0, "end": 1000},
                    {"type": "website", "start": 1002, "end": 2000, "url": "..."},
                    {"type": "linkedin", "start": 2002, "end": 3000, "url": "..."}
                ]
            }
    """
    parts = []
    source_info = {"sections": []}
    current_pos = 0
    
    # 1) Assignment-level description
    desc_text = f"{name} is a company in the '{primary_industry_classification}' industry. {business_description}"
    parts.append(desc_text)
    source_info["sections"].append({
        "type": "business_description",
        "start": current_pos,
        "end": current_pos + len(desc_text),
        "url": url or None
    })
    current_pos += len(desc_text) + 2  # +2 for "\n\n"
    
    # 2) Website – take top N pages (5-10)
    for page in pages[:10]:
        text = page.get("text", "")
        if text:
            page_url = page.get("url", url)
            parts.append(text)
            source_info["sections"].append({
                "type": "website",
                "start": current_pos,
                "end": current_pos + len(text),
                "url": page_url
            })
            current_pos += len(text) + 2  # +2 for "\n\n"
    
    # 3) LinkedIn – append posts talking about offerings/clients
    for post in posts[:20]:
        content = post.get("content", "")
        if content:
            post_url = post.get("postUrl") or post.get("url") or None
            parts.append(content)
            source_info["sections"].append({
                "type": "linkedin",
                "start": current_pos,
                "end": current_pos + len(content),
                "url": post_url
            })
            current_pos += len(content) + 2  # +2 for "\n\n"
    
    profile_text = "\n\n".join(parts)
    return profile_text, source_info


def _validate_evidence_quotes(extracted: Dict, profile_text: str) -> Dict:
    """
    Validate that evidence quotes are present in profile_text (exact or near-exact matches).
    Removes quotes that cannot be found in profile_text to prevent hallucinations.
    Returns the original quote if found, or finds the closest matching sentence from profile_text.
    
    Args:
        extracted: LLM-extracted dictionary with evidence field
        profile_text: Original profile text to validate against
        
    Returns:
        Validated extracted dictionary with evidence quotes that exist in profile_text
    """
    import re
    
    evidence = extracted.get("evidence", {})
    
    # Normalize profile_text for comparison (remove extra whitespace)
    profile_normalized = re.sub(r'\s+', ' ', profile_text).lower()
    profile_sentences = [s.strip() for s in re.split(r'[.!?]\s+', profile_text) if s.strip()]
    
    validated_evidence = {
        "business_activity": [],
        "products": [],
        "customer_segment": [],
        "product_mix": []
    }
    
    for category in ["business_activity", "products", "customer_segment", "product_mix"]:
        quotes = evidence.get(category, [])
        if not isinstance(quotes, list):
            continue
            
        for quote_entry in quotes:
            # Handle both old format (string) and new format (dict with quote, source, source_url)
            if isinstance(quote_entry, dict):
                quote = quote_entry.get("quote", "")
                source = quote_entry.get("source", "unknown")
                source_url = quote_entry.get("source_url")
            else:
                # Old format: just a string
                quote = quote_entry if isinstance(quote_entry, str) else str(quote_entry)
                source = None
                source_url = None
            
            if not quote or not quote.strip():
                continue
                
            quote_normalized = re.sub(r'\s+', ' ', quote).lower().strip()
            
            # Skip very short quotes (likely not meaningful)
            if len(quote_normalized) < 15:
                continue
                
            validated_quote = None
            
            # Strategy 1: Check if quote appears exactly in profile_text (case-insensitive, whitespace-normalized)
            if quote_normalized in profile_normalized:
                # Find the original sentence from profile_text that contains this quote
                quote_lower = quote.lower()
                found_original = None
                for orig_sentence in profile_sentences:
                    if quote_lower in orig_sentence.lower():
                        # Use the original sentence (preserves case, punctuation, exact wording)
                        found_original = orig_sentence
                        break
                
                if found_original:
                    validated_quote = found_original
                else:
                    # Fallback: use the quote as-is (it's in profile_text, just couldn't find original)
                    validated_quote = quote
            
            # Strategy 2: Check if quote is a substring of any sentence in profile_text
            if not validated_quote:
                quote_words = set(re.findall(r'\b\w{3,}\b', quote_normalized))  # Words of 3+ chars
                if len(quote_words) >= 3:  # Need at least 3 meaningful words
                    best_match = None
                    best_overlap = 0
                    
                    for orig_sentence in profile_sentences:
                        sentence_normalized = re.sub(r'\s+', ' ', orig_sentence).lower()
                        sentence_words = set(re.findall(r'\b\w{3,}\b', sentence_normalized))
                        
                        if len(sentence_words) == 0:
                            continue
                        
                        # Calculate word overlap ratio
                        overlap = len(quote_words & sentence_words)
                        overlap_ratio = overlap / len(quote_words) if len(quote_words) > 0 else 0
                        
                        # Require high overlap (90%+) and quote should be substantial part of sentence
                        if overlap_ratio >= 0.9 and overlap >= 3:
                            if overlap_ratio > best_overlap:
                                best_overlap = overlap_ratio
                                best_match = orig_sentence
                    
                    if best_match:
                        validated_quote = best_match
            
            # If quote was validated, add it (with source info if available)
            if validated_quote:
                if source or source_url:
                    # New format: include source tracking
                    validated_evidence[category].append({
                        "quote": validated_quote,
                        "source": source or "unknown",
                        "source_url": source_url
                    })
                else:
                    # Old format: just the quote string (for backward compatibility)
                    validated_evidence[category].append(validated_quote)
            # If no match found, skip this quote (it's likely a hallucination)
    
    extracted["evidence"] = validated_evidence
    return extracted


def extract_target_structured_llm(
    name: str,
    url: str,
    business_description: str,
    primary_industry_classification: str,
    profile_text: str,
    source_info: Dict = None,
    api_key: str = None
) -> Dict:
    """
    Call OpenAI LLM to extract structured target data with exact schema.
    
    Args:
        name: Company name
        url: Company URL
        business_description: Business description from assignment
        primary_industry_classification: Primary industry classification
        profile_text: Full profile text (business description + website + LinkedIn)
        source_info: Source mapping info (dict with sections mapping text positions to sources)
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var)
    
    Returns:
        Dict with: name, url, primary_industry_classification, business_description,
                   business_activity[], customer_segment[], product_mix{}, 
                   industries[], evidence{}, raw_profile_text
    """
    if not openai_available:
        raise ValueError("OpenAI package not installed. Install with: pip install openai")
    
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
{profile_text[:20000]}

Using ONLY the information in the profile text and assignment fields, extract the following JSON fields:

{{
  "name": "{name}",
  "url": "{url}",
  "primary_industry_classification": "{primary_industry_classification}",
  "business_description": "{business_description}",
  "business_activity": ["normalized product/service phrases"],
  "products": ["specific product names, solutions, or offerings mentioned in profile_text (e.g., 'Enterprise Health Record', 'Huron Research product suite', 'Revenue Cycle Managed Services')"],
  "customer_segment": ["normalized buyer/verticals"],
  "product_mix": {{"segment_name": weight_between_0_and_1}},
  "similar_industries": ["similar OWN industries - industries where comparable companies operate (e.g., for a consulting firm: Consulting Services, Business Services, Professional Services)"],
  "customer_industries": ["customer industry verticals served - industries where the company's customers operate (e.g., Healthcare, Education & Research, Financial Services)"],
  "evidence": {{
    "business_activity": [
      {{
        "quote": "EXACT quote from profile_text about products/services",
        "source": "business_description" or "website" or "linkedin",
        "source_url": "{url}" or "linkedin_post_url" or "website_page_url" or null
      }}
    ],
    "products": [
      {{
        "quote": "EXACT quote from profile_text mentioning specific product names, solutions, or offerings",
        "source": "business_description" or "website" or "linkedin",
        "source_url": "{url}" or "linkedin_post_url" or "website_page_url" or null
      }}
    ],
    "customer_segment": [
      {{
        "quote": "EXACT quote from profile_text about customers",
        "source": "business_description" or "website" or "linkedin",
        "source_url": "{url}" or "linkedin_post_url" or "website_page_url" or null
      }}
    ],
    "product_mix": [
      {{
        "quote": "EXACT quote from profile_text about segment distribution",
        "source": "business_description" or "website" or "linkedin",
        "source_url": "{url}" or "linkedin_post_url" or "website_page_url" or null
      }}
    ]
  }},
  "raw_profile_text": "{profile_text[:5000]}"
}}

CRITICAL RULES FOR EVIDENCE EXTRACTION:
- evidence.business_activity: Copy EXACT sentences/phrases from profile_text that mention products or services. Do NOT paraphrase, summarize, or create new text. Use the EXACT wording from profile_text, even if it's long. Include full sentences that contain product/service mentions.
- evidence.products: Copy EXACT sentences/phrases from profile_text that mention specific product names, solutions, software products, or offerings (e.g., product suite names, solution names, software platforms). Do NOT paraphrase or summarize. Use the EXACT wording from profile_text, even if it's long. Include full sentences that contain product name mentions.
- evidence.customer_segment: Copy EXACT sentences/phrases from profile_text that mention customer types, verticals, or buyer segments. Do NOT paraphrase or summarize. Use the EXACT wording from profile_text.
- evidence.product_mix: Copy EXACT sentences/phrases from profile_text that mention segment distribution, percentages, or revenue splits. Do NOT paraphrase or create new text. Use the EXACT wording from profile_text, including any numbers or percentages mentioned.
- If a quote is long, include the FULL quote. Do NOT truncate or shorten quotes.
- Each evidence quote MUST appear verbatim somewhere in the profile_text above.
- If you cannot find an exact quote in profile_text, use an empty array [] for that evidence category.
- Do NOT invent or hallucinate quotes that are not in the profile_text.

CRITICAL RULES FOR EVIDENCE SOURCE TRACKING:
- For each evidence quote, identify the SOURCE from where it came:
  * "business_description": If quote appears in the assignment business_description (first part of profile_text)
  * "website": If quote appears in website content (middle part of profile_text)
  * "linkedin": If quote appears in LinkedIn post content (last part of profile_text)
- source_url: If you can identify a specific URL (e.g., from website content mentioning a page, or if profile_text includes URLs), include it. Otherwise, use null or the base company URL "{url}".
- The source field helps track where each piece of evidence originated (assignment input vs crawled website vs LinkedIn posts).

Other Rules:
- Extract 3-7 most important business_activity items (normalized phrases describing types of activities/services)
- Extract 2-10 specific products (product names, solution names, software products, offerings mentioned in profile_text - e.g., "Enterprise Health Record", "Huron Research product suite", "Revenue Cycle Managed Services", "Enterprise Resource Planning consulting")
- Extract 2-5 main customer_segment types (normalized phrases)
- product_mix: Create a dictionary where keys are segment names (normalized, e.g., "healthcare_consulting") and values are weights between 0.0 and 1.0 that roughly sum to 1.0. Base weights on:
  * Explicit percentages mentioned in profile_text (e.g., "50% healthcare" → 0.5)
  * Frequency of segment mentions
  * Emphasis in text (e.g., "primary segment", "largest segment")
  * If no explicit distribution, estimate from text emphasis
- similar_industries: List similar OWN industries (industries where comparable companies operate). This is DIFFERENT from primary_industry_classification - it should include variations and related industry types (e.g., if primary_industry_classification is "Research and Consulting Services", similar_industries might be ["Consulting Services", "Business Services", "Professional Services", "Information Technology Services"]). Extract from profile_text if mentioned, or infer from primary_industry_classification.
- customer_industries: List customer industry verticals served (NOT the company's own industry, but industries where customers operate). For consulting firms, this means industries/verticals they serve (e.g., ["Healthcare", "Education & Research", "Financial Services"]). For software companies, this means industries their software serves. Extract from profile_text mentioning "industries we serve", "serving [industry]", etc.
- Do not invent segments that are not clearly suggested by the text
- Use clear, normalized names

CRITICAL DISTINCTION:
- similar_industries = OWN industries (what the company IS - for finding comparable companies)
- customer_industries = Customer industries (who the company SERVES - for scoring similarity)

Return ONLY the JSON object, no other text."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a structured data extraction assistant. Return only valid JSON. CRITICAL: Evidence quotes must be EXACT text from the profile_text provided, not paraphrased or invented."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=4000  # Increased to allow for longer evidence quotes
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    extracted = json.loads(response_text)
    
    # Validate evidence quotes are present in profile_text (exact or near-exact matches)
    # This removes any hallucinations and ensures all evidence quotes are from profile_text
    extracted = _validate_evidence_quotes(extracted, profile_text)
    
    # Log validation results
    evidence = extracted.get("evidence", {})
    total_quotes = sum(len(quotes) for quotes in evidence.values())
    if total_quotes == 0:
        print("⚠ Warning: No evidence quotes validated. All quotes may have been filtered out.")
    else:
        print(f"✓ Validated {total_quotes} evidence quote(s) from profile_text")
        for category, quotes in evidence.items():
            if quotes:
                print(f"  - {category}: {len(quotes)} quote(s)")
    
    # Post-process evidence to add source info if LLM didn't provide it
    # (backward compatibility and to ensure source tracking)
    evidence = extracted.get("evidence", {})
    if source_info:
        # Helper to find source for a quote based on position in profile_text
        def find_source_for_quote(quote: str) -> Dict:
            quote_lower = quote.lower().strip()
            profile_lower = profile_text.lower()
            quote_pos = profile_lower.find(quote_lower[:min(50, len(quote_lower))])
            
            if quote_pos == -1:
                # Try to find partial match
                for word in quote_lower.split()[:3]:
                    if len(word) > 5:
                        quote_pos = profile_lower.find(word)
                        if quote_pos != -1:
                            break
            
            if quote_pos != -1:
                # Find which section this position belongs to
                for section in source_info.get("sections", []):
                    if section["start"] <= quote_pos <= section["end"]:
                        return {
                            "source": section["type"],
                            "source_url": section.get("url")
                        }
            
            # Default: assume business_description if not found
            return {"source": "business_description", "source_url": url}
        
        # Update evidence entries if they're strings (old format) or missing source
        for category in ["business_activity", "products", "customer_segment", "product_mix"]:
            quotes = evidence.get(category, [])
            updated_quotes = []
            for quote_entry in quotes:
                if isinstance(quote_entry, str):
                    # Old format: just a string, convert to dict with source
                    source_dict = find_source_for_quote(quote_entry)
                    updated_quotes.append({
                        "quote": quote_entry,
                        "source": source_dict["source"],
                        "source_url": source_dict["source_url"]
                    })
                elif isinstance(quote_entry, dict):
                    # New format: already has quote, source, source_url
                    # But ensure source is set if missing
                    if "quote" not in quote_entry or "source" not in quote_entry:
                        if "quote" in quote_entry:
                            source_dict = find_source_for_quote(quote_entry["quote"])
                            updated_quotes.append({
                                "quote": quote_entry["quote"],
                                "source": quote_entry.get("source", source_dict["source"]),
                                "source_url": quote_entry.get("source_url", source_dict["source_url"])
                            })
                        else:
                            # Invalid format, skip
                            continue
                    else:
                        updated_quotes.append(quote_entry)
            
            evidence[category] = updated_quotes
        
        extracted["evidence"] = evidence
    
    # Add full raw_profile_text (not truncated)
    extracted["raw_profile_text"] = profile_text
    
    return extracted


def create_target_from_info(
    name: str,
    url: str,
    business_description: str,
    primary_industry_classification: str,
    linkedin_url: str = None,
    linkedin_company_name: str = None,
    months_back: int = 8,
    api_key: str = None
) -> Dict:
    """
    Create target.json from assignment inputs + website + LinkedIn.
    
    Args:
        name: Company name
        url: Homepage URL
        business_description: Brief business description from assignment
        primary_industry_classification: Primary industry (e.g., "Research and Consulting Services")
        linkedin_url: LinkedIn company URL (optional)
        linkedin_company_name: LinkedIn company name/handle (optional)
        months_back: Number of months of LinkedIn posts to fetch (default: 8)
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var)
    
    Returns:
        Dict with target.json structure including unique identifier and metadata
    
    Raises:
        ValueError: If required inputs missing or LLM extraction fails
    """
    print("="*80)
    print("Creating Target from Assignment Inputs + Website + LinkedIn")
    print("="*80)
    print(f"Name: {name}")
    print(f"URL: {url}")
    print(f"Industry: {primary_industry_classification}")
    print()
    
    # Step 1: Crawl website
    print("[1/4] Crawling website...")
    print(f"  URL: {url}")
    website_pages = fetch_website_content(url, max_pages=10)
    print(f"  ✓ Crawled {len(website_pages)} pages")
    
    # Filter relevant pages
    relevant_pages = filter_relevant_pages(website_pages)
    print(f"  ✓ Filtered to {len(relevant_pages)} relevant pages")
    print()
    
    # Step 2: Fetch LinkedIn posts (optional)
    posts = []
    linkedin_posts_fetched_count = 0
    
    if linkedin_company_name or linkedin_url:
        print(f"[2/4] Fetching LinkedIn posts (last {months_back} months)...")
        try:
            if linkedin_company_name:
                posts = fetch_linkedin_posts(company_name=linkedin_company_name, months_back=months_back)
            elif linkedin_url:
                posts = fetch_linkedin_posts(company_url=linkedin_url, months_back=months_back)
            
            # Store the count before filtering (for metadata)
            linkedin_posts_fetched_count = len(posts)
            
            # Check if any posts were found
            if linkedin_posts_fetched_count == 0:
                print(f"  ⚠ No LinkedIn posts found (company page may not exist or have no posts)")
                print(f"  Continuing without LinkedIn data...")
            else:
                print(f"  ✓ Fetched {linkedin_posts_fetched_count} posts from LinkedIn")
                
                # Filter relevant posts
                relevant_posts = filter_relevant_linkedin_posts(posts, months_back=months_back)
                print(f"  ✓ Found {len(relevant_posts)} relevant posts from last {months_back} months")
        except Exception as e:
            error_msg = str(e).lower()
            if "memory limit" in error_msg:
                print(f"  ⚠ Apify memory limit reached - skipping LinkedIn")
            elif "not found" in error_msg or "404" in error_msg or "company" in error_msg and "not exist" in error_msg:
                print(f"  ⚠ LinkedIn page not found - company may not have a LinkedIn page")
            elif "timeout" in error_msg:
                print(f"  ⚠ LinkedIn fetch timed out - skipping LinkedIn")
            else:
                print(f"  ⚠ LinkedIn fetch failed: {str(e)[:100]}")
            print(f"  Continuing without LinkedIn data...")
            posts = []  # Ensure posts is empty on error
            linkedin_posts_fetched_count = 0
    else:
        print(f"[2/4] LinkedIn: Skipped (no LinkedIn URL/name provided)")
        print(f"  ℹ LinkedIn is optional - continuing with website data only")
    
    # Initialize relevant_posts (may be empty if LinkedIn was skipped or failed)
    if 'relevant_posts' not in locals():
        relevant_posts = []
    
    print()
    
    # Step 3: Build profile_text with source tracking
    print("[3/4] Building profile text from all sources...")
    profile_text, source_info = build_target_profile_text(
        name=name,
        business_description=business_description,
        primary_industry_classification=primary_industry_classification,
        pages=relevant_pages,
        posts=relevant_posts,
        url=url
    )
    print(f"  ✓ Profile text length: {len(profile_text)} characters")
    print(f"  ✓ Source sections: {len(source_info.get('sections', []))}")
    print()
    
    # Step 4: Extract structured data using LLM (with source tracking)
    print("[4/4] Extracting structured data with LLM (including source tracking)...")
    extracted = extract_target_structured_llm(
        name=name,
        url=url,
        business_description=business_description,
        primary_industry_classification=primary_industry_classification,
        profile_text=profile_text,
        source_info=source_info,
        api_key=api_key
    )
    print(f"  ✓ Extracted:")
    print(f"    Business activities: {len(extracted.get('business_activity', []))}")
    print(f"    Products: {len(extracted.get('products', []))}")
    print(f"    Customer segments: {len(extracted.get('customer_segment', []))}")
    print(f"    Product mix segments: {len(extracted.get('product_mix', {}))}")
    print()
    
    # Step 5: Build final target.json with unique identifier and metadata
    target_id = str(uuid.uuid4())
    
    # Build target dict with explicit field selection
    # CRITICAL: We explicitly select fields - we do NOT copy all extracted fields
    # This ensures old fields like 'industries' are NOT saved
    target = {
        # Assignment fields (from LLM extraction)
        "name": extracted.get("name", name),
        "url": extracted.get("url", url),
        "primary_industry_classification": extracted.get("primary_industry_classification", primary_industry_classification),
        "business_description": extracted.get("business_description", business_description),
        
        # Extracted fields - explicitly selected
        "business_activity": extracted.get("business_activity", []),
        "products": extracted.get("products", []),
        "customer_segment": extracted.get("customer_segment", []),
        "product_mix": extracted.get("product_mix", {}),
        "similar_industries": extracted.get("similar_industries", []),
        "customer_industries": extracted.get("customer_industries", extracted.get("industries", [])),  # Backward compatibility: fallback to old 'industries' field
        # NOTE: We do NOT save 'industries' field - only 'customer_industries' (with fallback)
        "evidence": extracted.get("evidence", {}),
        "raw_profile_text": extracted.get("raw_profile_text", profile_text),
        
        # Unique identifier and metadata
        "target_id": target_id,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "created_by": "create_target_from_info.py",
            "version": "1.0",
            "extraction_method": "llm_openai_gpt4o",
            "source": "website_linkedin",
            "website_pages_crawled": len(website_pages),
            "website_pages_relevant": len(relevant_pages),
            "linkedin_posts_fetched": linkedin_posts_fetched_count,
            "linkedin_posts_relevant": len(relevant_posts),
            "profile_text_length": len(profile_text)
        }
    }
    
    # Ensure old 'industries' field is NOT in target (if it exists, it was copied from extracted somehow)
    # This is defensive - we explicitly don't save it above, but just in case
    if 'industries' in target:
        del target['industries']
    
    return target


def create_target_from_args():
    """Create target.json from command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create target.json from assignment inputs + website + LinkedIn',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires OPENAI_API_KEY and APIFY_API_TOKEN):
  python create_target_from_info.py \\
    --name "Huron Consulting Group Inc." \\
    --url "http://www.huronconsultinggroup.com/" \\
    --description "Huron Consulting Group Inc. provides consultancy..." \\
    --primary-industry-classification "Research and Consulting Services"
  
  # With LinkedIn:
  python create_target_from_info.py \\
    --name "Company Name" \\
    --url "https://company.com" \\
    --description "Business description..." \\
    --primary-industry-classification "Industry Name" \\
    --linkedin "company-name" \\
    --linkedin-url "https://linkedin.com/company/company-name"
  
  # Alternative: using --industry (shorthand):
  python create_target_from_info.py \\
    --name "Company Name" \\
    --url "https://company.com" \\
    --description "Business description..." \\
    --industry "Industry Name"
        """
    )
    parser.add_argument('--name', type=str, required=True, help='Company name (required)')
    parser.add_argument('--url', type=str, required=True, help='Homepage URL (required)')
    parser.add_argument('--description', type=str, required=True, help='Business description (required)')
    parser.add_argument('--primary-industry-classification', '--industry', type=str, required=True, 
                        dest='primary_industry_classification',
                        help='Primary industry classification (required, e.g., "Research and Consulting Services")')
    parser.add_argument('--linkedin', type=str, default=None, 
                        help='LinkedIn company name/handle (optional - skip if company has no LinkedIn page)')
    parser.add_argument('--linkedin-url', type=str, default=None, 
                        help='LinkedIn company URL (optional - skip if company has no LinkedIn page)')
    parser.add_argument('--months-back', type=int, default=8, help='Months of LinkedIn posts to fetch (default: 8)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: comps/data/target_{companyname}.json)')
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI API key (or use OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Create target.json
    print("="*80)
    print("Creating target.json from assignment inputs + website + LinkedIn")
    print("="*80)
    print(f"Name: {args.name}")
    print(f"URL: {args.url}")
    print(f"Primary Industry Classification: {args.primary_industry_classification}")
    print(f"Description: {args.description[:100]}..." if len(args.description) > 100 else f"Description: {args.description}")
    if args.linkedin or args.linkedin_url:
        print(f"LinkedIn: {args.linkedin or args.linkedin_url}")
    print()
    
    target = create_target_from_info(
        name=args.name,
        url=args.url,
        business_description=args.description,
        primary_industry_classification=args.primary_industry_classification,
        linkedin_company_name=args.linkedin,
        linkedin_url=args.linkedin_url,
        months_back=args.months_back,
        api_key=args.api_key
    )
    
    # Generate filename from company name if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        # Create filename from company name: target_{companyname}.json
        # Sanitize company name for filename (remove special chars, spaces -> underscores)
        import re
        company_name_safe = re.sub(r'[^a-zA-Z0-9_-]', '', args.name.replace(' ', '_').replace('.', '').replace(',', ''))
        company_name_safe = company_name_safe.lower()[:50]  # Limit length
        
        # Determine correct path: script is in comps/data/, so write to same directory
        script_dir = Path(__file__).parent  # comps/data/
        output_path = script_dir / f"target_{company_name_safe}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(target, f, indent=2)
    
    print("="*80)
    print(f"✓ Created target file: {output_path}")
    print(f"  Filename format: target_{{companyname}}.json (for traceability and caching)")
    print("="*80)
    print(json.dumps(target, indent=2, default=str)[:1000] + "...")
    print()
    print(f"Target ID: {target['target_id']}")
    print(f"Business activities: {len(target['business_activity'])}")
    print(f"Products: {len(target.get('products', []))}")
    print(f"Customer segments: {len(target['customer_segment'])}")
    print(f"Product mix segments: {len(target['product_mix'])}")


if __name__ == "__main__":
    create_target_from_args()
