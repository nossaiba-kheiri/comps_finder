#!/usr/bin/env python3
"""
view_extraction.py: View LLM extraction results for a candidate company.

Usage:
    python view_extraction.py TICKER
    
Example:
    python view_extraction.py NDSN
    python view_extraction.py FCN
    python view_extraction.py KAI
"""

import pickle
import json
import sys
from pathlib import Path
from datetime import datetime


def view_extraction(ticker):
    """View LLM extraction for a given ticker."""
    cache_dir = Path("data/cache/llm_extraction")
    
    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        print("   Run the pipeline first to generate extractions.")
        return
    
    # Find all files for this ticker
    pkl_files = list(cache_dir.glob(f"{ticker.upper()}_*.pkl"))
    
    if not pkl_files:
        print(f"❌ No extraction found for {ticker.upper()}")
        print(f"   Searched in: {cache_dir}")
        print(f"   Pattern: {ticker.upper()}_*.pkl")
        print("\n💡 Available extractions:")
        all_pkls = list(cache_dir.glob("*.pkl"))
        if all_pkls:
            # Extract tickers from filenames
            tickers_found = set()
            for pkl in all_pkls[:20]:  # Show first 20
                parts = pkl.stem.split('_')
                if parts:
                    tickers_found.add(parts[0])
            for t in sorted(tickers_found)[:10]:
                print(f"   - {t}")
        return
    
    # Load the most recent one
    pkl_file = sorted(pkl_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    # Load extraction data
    try:
        with open(pkl_file, 'rb') as f:
            cached_data = pickle.load(f)
            extracted = cached_data.get('extracted', {})
    except Exception as e:
        print(f"❌ Error loading extraction file: {e}")
        return
    
    # Load metadata
    metadata = {}
    metadata_file = pkl_file.with_suffix('').with_name(pkl_file.stem + '_metadata.json')
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except:
            pass
    
    # Display extraction
    print(f"\n{'='*80}")
    print(f"📊 LLM EXTRACTION FOR {ticker.upper()}")
    print(f"{'='*80}\n")
    
    # Metadata
    if metadata:
        created_at = metadata.get('created_at', '')
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                print(f"🕐 Cached: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                pass
        print(f"📝 Prompt Version: {metadata.get('prompt_version', 'N/A')}")
        print()
    
    # Business Model
    print("📋 BUSINESS MODEL:")
    print(f"  Type: {extracted.get('business_model_type', 'N/A')}")
    print(f"  Services Share: {extracted.get('services_share_estimate', 'N/A')}")
    
    # Economic Signature
    print("\n💰 ECONOMIC SIGNATURE:")
    econ_sig = extracted.get('economic_signature', {})
    if econ_sig and isinstance(econ_sig, dict):
        print(f"  Capital Equipment Share: {econ_sig.get('capital_equipment_share', 0):.1%}")
        print(f"  Aftermarket Service Share: {econ_sig.get('aftermarket_service_share', 0):.1%}")
        print(f"  Consumables Share: {econ_sig.get('consumables_share', 0):.1%}")
        print(f"  Software Recurring Share: {econ_sig.get('software_recurring_share', 0):.1%}")
        print(f"  Project Services Share: {econ_sig.get('project_services_share', 0):.1%}")
        print(f"  IP Intensity: {econ_sig.get('ip_intensity', 0):.2f} (0=commodity, 1=highly proprietary)")
        print(f"  Customer Lock-in: {econ_sig.get('customer_lock_in', 0):.2f} (0=low, 1=very high)")
        print(f"  Replacement Cycle: {econ_sig.get('replacement_cycle_years', 0):.1f} years")
        print(f"  Gross Margin Tier: {econ_sig.get('gross_margin_tier', 'N/A')}")
        print(f"  Asset Intensity: {econ_sig.get('asset_intensity', 0):.2f} (0=light, 1=heavy)")
    else:
        print("  ⚠️  Not extracted by LLM - will be inferred from revenue_channels")
    
    # Business Activity
    print("\n🏭 BUSINESS ACTIVITY:")
    activities = extracted.get('business_activity', [])
    if activities:
        for activity in activities[:10]:
            print(f"  - {activity}")
        if len(activities) > 10:
            print(f"  ... and {len(activities) - 10} more")
    else:
        print("  (None)")
    
    # Customer Segment
    print("\n👥 CUSTOMER SEGMENT:")
    segments = extracted.get('customer_segment', [])
    if segments:
        for segment in segments[:10]:
            print(f"  - {segment}")
        if len(segments) > 10:
            print(f"  ... and {len(segments) - 10} more")
    else:
        print("  (None)")
    
    # Revenue Channels
    print("\n💵 REVENUE CHANNELS (Top 5):")
    channels = extracted.get('revenue_channels', {})
    if channels and isinstance(channels, dict):
        sorted_channels = sorted(channels.items(), key=lambda x: x[1] or 0, reverse=True)
        for channel, share in sorted_channels[:5]:
            share_val = share or 0
            if share_val > 0:
                print(f"  - {channel}: {share_val:.1%}")
    else:
        print("  (None)")
    
    # Revenue Archetypes
    print("\n🎯 REVENUE ARCHETYPES:")
    archetypes = extracted.get('revenue_archetypes', {})
    if archetypes and isinstance(archetypes, dict):
        for arch, share in sorted(archetypes.items(), key=lambda x: x[1] or 0, reverse=True):
            share_val = share or 0
            if share_val > 0:
                print(f"  - {arch}: {share_val:.1%}")
    else:
        print("  (None)")
    
    # Full JSON option
    if '--full' in sys.argv or '-f' in sys.argv:
        print(f"\n{'='*80}")
        print("📄 FULL EXTRACTION (JSON):")
        print(f"{'='*80}\n")
        print(json.dumps(extracted, indent=2, default=str, ensure_ascii=False))
    else:
        print(f"\n💡 Tip: Use --full or -f to see complete JSON output")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_extraction.py TICKER [--full]")
        print("\nExample:")
        print("  python view_extraction.py NDSN")
        print("  python view_extraction.py FCN --full")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    view_extraction(ticker)

