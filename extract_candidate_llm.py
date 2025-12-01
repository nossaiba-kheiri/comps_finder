#!/usr/bin/env python3
"""
extract_candidate_llm.py: Extract LLM extraction data for a candidate from ranked CSV.

Usage:
    python extract_candidate_llm.py TICKER [--csv-file CSV_FILE]
    
Example:
    python extract_candidate_llm.py KAI
    python extract_candidate_llm.py NDSN --csv-file data/outputs/husky_technologies_ranked.csv
"""

import csv
import json
import sys
import ast
from pathlib import Path


def extract_candidate_llm_data(ticker, csv_file="data/outputs/husky_technologies_ranked.csv"):
    """
    Extract full LLM extraction data for a specific candidate from ranked CSV.
    
    Args:
        ticker: Company ticker symbol
        csv_file: Path to ranked CSV file
    
    Returns:
        Dict with LLM extraction data or None if not found
    """
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('ticker', '').upper() == ticker.upper():
                # Parse segment_mix if it's a string
                segment_mix_str = row.get('segment_mix', '{}')
                try:
                    if segment_mix_str and segment_mix_str != '{}':
                        segment_mix = ast.literal_eval(segment_mix_str)
                    else:
                        segment_mix = {}
                except:
                    segment_mix = {}
                
                # Parse initiatives if available
                initiatives_str = row.get('initiatives', '[]')
                try:
                    if initiatives_str and initiatives_str.startswith('['):
                        initiatives = ast.literal_eval(initiatives_str)
                    else:
                        initiatives = []
                except:
                    initiatives = []
                
                # Parse business_activity and customer_segment (comma-separated strings)
                business_activity = []
                if row.get('business_activity'):
                    business_activity = [x.strip() for x in row.get('business_activity', '').split(',') if x.strip()]
                
                customer_segment = []
                if row.get('customer_segment'):
                    customer_segment = [x.strip() for x in row.get('customer_segment', '').split(',') if x.strip()]
                
                revenue_model = []
                if row.get('revenue_model'):
                    revenue_model = [x.strip() for x in row.get('revenue_model', '').split(',') if x.strip()]
                
                extraction = {
                    'ticker': row.get('ticker', ''),
                    'name': row.get('name', ''),
                    'business_activity': business_activity,
                    'customer_segment': customer_segment,
                    'business_model_type': row.get('business_model_type', ''),
                    'services_share_estimate': float(row.get('services_share_estimate', 0) or 0),
                    'revenue_model': revenue_model,
                    'has_professional_services': row.get('has_professional_services', '').lower() == 'true',
                    'has_managed_services': row.get('has_managed_services', '').lower() == 'true',
                    'has_software_product': row.get('has_software_product', '').lower() == 'true',
                    'segment_mix': segment_mix,
                    'initiatives': initiatives,
                    'evidence_urls': row.get('evidence_urls', ''),
                    'evidence_quotes': row.get('evidence_quotes', ''),
                    'LLM_confidence': row.get('LLM_confidence', ''),
                    'prompt_version': row.get('prompt_version', ''),
                }
                
                return extraction
    
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_candidate_llm.py TICKER [--csv-file CSV_FILE]")
        print("\nExample:")
        print("  python extract_candidate_llm.py KAI")
        print("  python extract_candidate_llm.py NDSN --csv-file data/outputs/husky_technologies_ranked.csv")
        sys.exit(1)
    
    ticker = sys.argv[1]
    csv_file = "data/outputs/husky_technologies_ranked.csv"
    
    # Check for --csv-file argument
    if '--csv-file' in sys.argv:
        idx = sys.argv.index('--csv-file')
        if idx + 1 < len(sys.argv):
            csv_file = sys.argv[idx + 1]
    
    extraction = extract_candidate_llm_data(ticker, csv_file)
    
    if extraction:
        print("="*80)
        print(f"📊 LLM EXTRACTION FOR: {extraction['ticker']} - {extraction['name']}")
        print("="*80)
        print()
        
        print("📋 BUSINESS MODEL:")
        print(f"  Type: {extraction['business_model_type']}")
        print(f"  Services Share: {extraction['services_share_estimate']:.1%}")
        print(f"  Has Professional Services: {extraction['has_professional_services']}")
        print(f"  Has Managed Services: {extraction['has_managed_services']}")
        print(f"  Has Software Product: {extraction['has_software_product']}")
        
        print("\n🏭 BUSINESS ACTIVITY:")
        if extraction['business_activity']:
            for activity in extraction['business_activity']:
                print(f"  - {activity}")
        else:
            print("  (empty)")
        
        print("\n👥 CUSTOMER SEGMENT:")
        if extraction['customer_segment']:
            for segment in extraction['customer_segment']:
                print(f"  - {segment}")
        else:
            print("  (empty)")
        
        print("\n💵 REVENUE MODEL:")
        if extraction['revenue_model']:
            for model in extraction['revenue_model']:
                print(f"  - {model}")
        else:
            print("  (empty)")
        
        if extraction.get('segment_mix'):
            print("\n📊 SEGMENT MIX:")
            for segment, weight in extraction['segment_mix'].items():
                print(f"  - {segment}: {weight:.1%}")
        
        if extraction.get('initiatives'):
            print("\n🚀 INITIATIVES:")
            for init in extraction['initiatives'][:5]:  # Show first 5
                if isinstance(init, dict):
                    print(f"  - {init.get('name', 'N/A')}: {init.get('description', 'N/A')[:60]}...")
        
        if extraction.get('LLM_confidence'):
            print(f"\n🎯 LLM CONFIDENCE: {extraction['LLM_confidence']}")
        
        print("\n" + "="*80)
        print("\n📄 FULL JSON:")
        print("="*80 + "\n")
        print(json.dumps(extraction, indent=2, default=str))
    else:
        print(f"❌ No LLM extraction found for ticker: {ticker}")
        print(f"   Searched in: {csv_file}")


if __name__ == '__main__':
    main()

