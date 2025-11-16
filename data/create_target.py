#!/usr/bin/env python3
"""
create_target.py: Interactive script to create target.json
"""
import json
import os
from pathlib import Path

def create_target_interactive():
    """Interactively create target.json"""
    print("="*80)
    print("Target JSON Creator")
    print("="*80)
    print()
    
    target = {}
    
    # Name (required)
    print("1. Company Name (required)")
    name = input("   Enter company name: ").strip()
    if not name:
        print("   Error: Name is required!")
        return
    target["name"] = name
    print()
    
    # URL (optional)
    print("2. Company URL (optional)")
    url = input("   Enter company URL (or press Enter to skip): ").strip()
    if url:
        target["url"] = url
    print()
    
    # Products (required)
    print("3. Products/Services (required)")
    print("   Enter products one per line (empty line to finish):")
    products = []
    while True:
        product = input("   > ").strip()
        if not product:
            break
        products.append(product)
    if not products:
        print("   Error: At least one product is required!")
        return
    target["products"] = products
    print(f"   ✓ Added {len(products)} products")
    print()
    
    # Customers (required)
    print("4. Customer Types (required)")
    print("   Enter customer types one per line (empty line to finish):")
    customers = []
    while True:
        customer = input("   > ").strip()
        if not customer:
            break
        customers.append(customer)
    if not customers:
        print("   Error: At least one customer type is required!")
        return
    target["customers"] = customers
    print(f"   ✓ Added {len(customers)} customer types")
    print()
    
    # Segment Mix (optional)
    print("5. Segment Mix (optional)")
    print("   Enter segment revenue breakdown (press Enter to skip)")
    segment_mix = {}
    segments = ["Healthcare", "Education", "Commercial", "Public", "Financial Services", "Retail"]
    print("   Available segments:", ", ".join(segments))
    while True:
        segment = input("   Enter segment name (or press Enter to finish): ").strip()
        if not segment:
            break
        if segment not in segments:
            print(f"   Warning: '{segment}' not in canonical segments, but continuing...")
        try:
            weight = float(input(f"   Enter weight for {segment} (0.0-1.0): ").strip())
            if weight < 0 or weight > 1:
                print("   Error: Weight must be between 0.0 and 1.0!")
                continue
            segment_mix[segment] = weight
        except ValueError:
            print("   Error: Invalid weight!")
            continue
    if segment_mix:
        total = sum(segment_mix.values())
        if abs(total - 1.0) > 0.01:
            print(f"   Warning: Segment weights sum to {total}, not 1.0")
            normalize = input("   Normalize to 1.0? (y/n): ").strip().lower()
            if normalize == 'y':
                segment_mix = {k: v/total for k, v in segment_mix.items()}
                print("   ✓ Normalized segment mix")
        target["segment_mix"] = segment_mix
    print()
    
    # Text Profile (required)
    print("6. Text Profile (required)")
    print("   Enter a 2-3 sentence description combining products and customers:")
    print("   (Multi-line: press Enter for new line, empty line to finish)")
    text_profile_lines = []
    while True:
        line = input("   > ").strip()
        if not line and text_profile_lines:
            break
        if line:
            text_profile_lines.append(line)
    if not text_profile_lines:
        print("   Error: Text profile is required!")
        return
    target["text_profile"] = " ".join(text_profile_lines)
    print(f"   ✓ Added text profile ({len(target['text_profile'])} chars)")
    print()
    
    # Mode (optional)
    print("7. Mode (optional, default: 'all_segments')")
    mode = input("   Enter mode (or press Enter for default): ").strip()
    if mode:
        target["mode"] = mode
    else:
        target["mode"] = "all_segments"
    print()
    
    # Industry (optional)
    print("8. Industry (optional)")
    industry = input("   Enter industry (or press Enter to skip): ").strip()
    if industry:
        target["industry"] = industry
    print()
    
    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    print(json.dumps(target, indent=2))
    print()
    
    # Save
    save = input("Save to comps/data/target.json? (y/n): ").strip().lower()
    if save == 'y':
        output_path = Path("comps/data/target.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(target, f, indent=2)
        print(f"✓ Saved to {output_path}")
    else:
        print("Not saved.")


if __name__ == "__main__":
    create_target_interactive()

