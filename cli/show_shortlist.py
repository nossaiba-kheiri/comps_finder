#!/usr/bin/env python3
"""
Quick script to show the top 80 shortlist without running the full pipeline.
Just generates candidates and shows the shortlist.
"""
import os
import sys
import json
import argparse
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Import modules
from universe.generate_candidates import generate_candidates

# Directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_target(target_path):
    """Load target JSON."""
    with open(target_path, 'r') as f:
        return json.load(f)

def load_config():
    """Load runtime configuration."""
    config_path = os.path.join(PROJECT_ROOT, 'config', 'runtime.yaml')
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(
        description='Show top 80 shortlist without running full pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--target', type=str, required=True,
                        help='Path to target.json file')
    parser.add_argument('--openai', action='store_true',
                        help='Use real OpenAI embeddings (default: False, uses mock)')
    parser.add_argument('--all', action='store_true',
                        help='Show all 80 companies (default: shows top 20)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SHORTLIST PREVIEW (No Pipeline Run)")
    print("="*80)
    
    # Load target and config
    print("\n[1/2] Loading target and config...")
    target = load_target(args.target)
    config = load_config()
    target_id = target.get('name', 'target').replace(' ', '_').lower()
    print(f"✓ Target: {target.get('name')}")
    
    # Generate candidates
    print("\n[2/2] Generating candidates and creating shortlist...")
    candidates_df = generate_candidates(target, config, run_with_openai=args.openai)
    print(f"✓ Generated {len(candidates_df)} total candidates")
    
    # Create shortlist
    shortlist_cap = config.get('shortlist_cap', 80)
    shortlist_df = candidates_df.head(shortlist_cap).copy()
    print(f"✓ Shortlisted {len(shortlist_df)} candidates")
    
    # Display shortlist
    print("\n" + "="*80)
    print(f"TOP {len(shortlist_df)} SHORTLIST")
    print("="*80)
    
    display_count = len(shortlist_df) if args.all else min(20, len(shortlist_df))
    
    print(f"\n📊 Top {display_count} companies:")
    print(f"{'Rank':<6} {'Ticker':<8} {'Name':<50} {'rank_key':<10} {'Paths':<10} {'Industry':<30}")
    print("-" * 120)
    
    for idx, (i, row) in enumerate(shortlist_df.head(display_count).iterrows(), 1):
        ticker = row.get('ticker', 'N/A')
        name = str(row.get('name', 'N/A'))[:48]
        rank_key = row.get('rank_key', 0.0)
        paths = str(row.get('paths', ''))[:8]
        industry = str(row.get('industry', 'N/A'))[:28]
        print(f"{idx:<6} {ticker:<8} {name:<50} {rank_key:<10.3f} {paths:<10} {industry:<30}")
    
    if not args.all and len(shortlist_df) > 20:
        print(f"\n... and {len(shortlist_df) - 20} more companies")
        print(f"   (Use --all to see all {len(shortlist_df)} companies)")
    
    print("="*80)
    print("\n✅ Done! This was a quick preview - no evidence gathering or LLM calls.")
    print("   To run the full pipeline, use: python cli/run_pipeline.py --target <target.json> --openai")

if __name__ == '__main__':
    main()

