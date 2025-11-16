#!/usr/bin/env python3
"""
deduplicate_cik.py: Remove duplicate CIKs from universe_us.csv, keeping one ticker per CIK.

Strategy:
- Keep the "primary" ticker (shortest, or without suffixes like -N, -A, -B, etc.)
- Prefer tickers without special suffixes
- If all have suffixes, keep the shortest one
"""
import os
import sys
import pandas as pd

UNIVERSE_PATH = os.path.join(os.path.dirname(__file__), '../../data/universe_us.csv')


def select_primary_ticker(group):
    """
    Select the primary ticker from a group of tickers with the same CIK.
    Strategy:
    1. Prefer tickers without suffixes (-N, -A, -B, -O, -W, -WT, etc.)
    2. Prefer shorter tickers
    3. If same length, prefer alphabetically first
    """
    tickers = group['ticker'].tolist()
    
    # Check for suffixes (common patterns: -N, -A, -B, -O, -W, -WT, etc.)
    def has_suffix(ticker):
        ticker_str = str(ticker).upper()
        # Common suffix patterns
        if '-' in ticker_str:
            return True
        # Suffixes like: N, A, B, C, O, W, WT, R, U, etc. at the end
        if len(ticker_str) > 4:
            # Check if ends with common suffix patterns
            suffix_patterns = ['W', 'WT', 'R', 'U', 'N', 'A', 'B', 'C', 'O', 'P', 'Z']
            if any(ticker_str.endswith(p) for p in suffix_patterns):
                # Make sure it's not just a short ticker (like "A" or "AA")
                if len(ticker_str) <= 5:
                    return False
                return True
        return False
    
    # Separate into with/without suffixes
    no_suffix = [t for t in tickers if not has_suffix(str(t))]
    with_suffix = [t for t in tickers if has_suffix(str(t))]
    
    # Prefer no suffix
    if no_suffix:
        candidates = no_suffix
    else:
        candidates = with_suffix
    
    # Among candidates, prefer shortest, then alphabetical
    candidates_str = [str(t).upper() for t in candidates]
    primary = min(candidates_str, key=lambda x: (len(x), x))
    
    # Find the original case version
    for t in candidates:
        if str(t).upper() == primary:
            return group[group['ticker'] == t].iloc[0]
    
    # Fallback: return first
    return group.iloc[0]


def deduplicate_by_cik(df):
    """
    Remove duplicate CIKs, keeping one ticker per CIK.
    Returns deduplicated DataFrame.
    """
    print(f"  Before deduplication: {len(df):,} records, {df['cik'].nunique():,} unique CIKs")
    
    # Group by CIK and select primary ticker
    deduplicated = df.groupby('cik', group_keys=False).apply(select_primary_ticker).reset_index(drop=True)
    
    print(f"  After deduplication: {len(deduplicated):,} records, {deduplicated['cik'].nunique():,} unique CIKs")
    print(f"  Removed {len(df) - len(deduplicated):,} duplicate records")
    
    return deduplicated


def main(backup=True):
    """
    Deduplicate universe_us.csv by CIK.
    
    Args:
        backup: If True, create a backup before modifying
    """
    if not os.path.exists(UNIVERSE_PATH):
        print(f"Error: {UNIVERSE_PATH} not found")
        return
    
    # Load CSV
    print(f"Loading {UNIVERSE_PATH}...")
    df = pd.read_csv(UNIVERSE_PATH)
    df = df.fillna('')
    
    # Create backup
    if backup:
        backup_path = UNIVERSE_PATH + '.backup'
        print(f"Creating backup: {backup_path}")
        df.to_csv(backup_path, index=False)
    
    # Deduplicate
    print("\nDeduplicating by CIK...")
    df_dedup = deduplicate_by_cik(df)
    
    # Show examples of what was removed
    print("\nExamples of removed duplicates:")
    cik_counts = df.groupby('cik')['ticker'].count()
    duplicate_ciks = cik_counts[cik_counts > 1].head(5)
    for cik in duplicate_ciks.index:
        all_tickers = df[df['cik'] == cik]['ticker'].tolist()
        kept_ticker = df_dedup[df_dedup['cik'] == cik]['ticker'].iloc[0] if len(df_dedup[df_dedup['cik'] == cik]) > 0 else None
        removed = [t for t in all_tickers if t != kept_ticker]
        print(f"  CIK {cik}: Kept '{kept_ticker}', Removed: {removed}")
    
    # Save deduplicated CSV
    print(f"\nSaving deduplicated CSV to {UNIVERSE_PATH}...")
    df_dedup = df_dedup.sort_values('ticker').reset_index(drop=True)
    df_dedup.to_csv(UNIVERSE_PATH, index=False)
    print(f"✓ Saved {len(df_dedup):,} records (removed {len(df) - len(df_dedup):,} duplicates)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Deduplicate universe_us.csv by CIK')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')
    args = parser.parse_args()
    main(backup=not args.no_backup)

