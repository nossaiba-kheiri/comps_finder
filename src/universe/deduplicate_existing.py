#!/usr/bin/env python3
"""
Deduplicate existing universe_us.csv by CIK.
Keeps one ticker per CIK (prefers primary ticker without suffixes).
"""
import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNIVERSE_PATH = os.path.join(os.path.dirname(__file__), '../../data/universe_us.csv')


def select_primary_ticker(group):
    """Select primary ticker from group with same CIK."""
    tickers = group['ticker'].tolist()
    
    def has_suffix(t):
        t_str = str(t).upper()
        if '-' in t_str:
            return True
        if len(t_str) > 4:
            suffixes = ['W', 'WT', 'R', 'U', 'N', 'A', 'B', 'C', 'O', 'P', 'Z']
            if any(t_str.endswith(s) for s in suffixes):
                return len(t_str) > 5
        return False
    
    no_suffix = [t for t in tickers if not has_suffix(str(t))]
    candidates = no_suffix if no_suffix else tickers
    primary = min([str(t).upper() for t in candidates], key=lambda x: (len(x), x))
    
    for t in candidates:
        if str(t).upper() == primary:
            return group[group['ticker'] == t].iloc[0]
    return group.iloc[0]


def main():
    print("="*60)
    print("Deduplicating universe_us.csv by CIK")
    print("="*60)
    
    if not os.path.exists(UNIVERSE_PATH):
        print(f"Error: {UNIVERSE_PATH} not found")
        return
    
    # Load CSV
    print(f"\nLoading {UNIVERSE_PATH}...")
    df = pd.read_csv(UNIVERSE_PATH)
    df = df.fillna('')
    
    print(f"  Before: {len(df):,} records")
    print(f"  Unique CIKs: {df['cik'].nunique():,}")
    print(f"  Unique tickers: {df['ticker'].nunique():,}")
    
    # Count duplicates
    cik_counts = df.groupby('cik')['ticker'].count()
    duplicate_ciks = cik_counts[cik_counts > 1]
    print(f"  CIKs with multiple tickers: {len(duplicate_ciks):,}")
    print(f"  Duplicate records to remove: {duplicate_ciks.sum() - len(duplicate_ciks):,}")
    
    # Show examples
    print("\n  Examples of duplicates:")
    for cik in list(duplicate_ciks.head(5).index):
        tickers = df[df['cik'] == cik]['ticker'].tolist()
        name = df[df['cik'] == cik]['name'].iloc[0][:50] if len(df[df['cik'] == cik]) > 0 else 'N/A'
        print(f"    CIK {cik}: {tickers} ({name}...)")
    
    # Deduplicate
    print("\n  Deduplicating...")
    df_dedup = df.groupby('cik', group_keys=False).apply(select_primary_ticker).reset_index(drop=True)
    
    print(f"  After: {len(df_dedup):,} records")
    print(f"  Unique CIKs: {df_dedup['cik'].nunique():,}")
    print(f"  Removed: {len(df) - len(df_dedup):,} duplicate records")
    
    # Show what was kept/removed for examples
    print("\n  Examples of kept/removed:")
    for cik in list(duplicate_ciks.head(3).index):
        all_tickers = df[df['cik'] == cik]['ticker'].tolist()
        kept_ticker = df_dedup[df_dedup['cik'] == cik]['ticker'].iloc[0] if len(df_dedup[df_dedup['cik'] == cik]) > 0 else None
        removed = [t for t in all_tickers if t != kept_ticker]
        print(f"    CIK {cik}: Kept '{kept_ticker}', Removed: {removed}")
    
    # Create backup
    backup_path = UNIVERSE_PATH + '.backup'
    print(f"\n  Creating backup: {backup_path}")
    df.to_csv(backup_path, index=False)
    
    # Save deduplicated CSV
    df_dedup = df_dedup.sort_values('ticker').reset_index(drop=True)
    print(f"  Saving deduplicated CSV...")
    df_dedup.to_csv(UNIVERSE_PATH, index=False)
    
    print(f"\n✓ Done! Deduplicated CSV saved to {UNIVERSE_PATH}")
    print(f"  Backup saved to {backup_path}")


if __name__ == "__main__":
    main()

