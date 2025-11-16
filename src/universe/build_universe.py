import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import sqlite3
import yfinance as yf

SEC_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_DB = Path(os.path.dirname(__file__)) / "../../data/yf_desc_cache.sqlite"
CACHE_TTL_HOURS = 24
OUT_PATH = os.path.join(os.path.dirname(__file__), '../../data/universe_us.csv')

def _init_cache():
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    # Check if table exists and what columns it has
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='desc_cache'")
    table_exists = cur.fetchone() is not None
    
    if table_exists:
        # Check if old schema (with 'description' column) exists
        cur.execute("PRAGMA table_info(desc_cache)")
        columns = [col[1] for col in cur.fetchall()]
        if 'description' in columns and 'blob' not in columns:
            # Migrate old schema to new schema
            print("Migrating cache schema from 'description' to 'blob'...")
            cur.execute("""
                CREATE TABLE desc_cache_new (
                    ticker TEXT PRIMARY KEY,
                    blob TEXT,
                    ts TEXT
                )
            """)
            # Copy old data, converting description to JSON blob format
            cur.execute("SELECT ticker, description, ts FROM desc_cache")
            for row in cur.fetchall():
                ticker, desc, ts = row
                blob_dict = {"summary": desc or "", "sector": "", "industry": "", "country": "", "website": "", "exchange": ""}
                cur.execute("INSERT INTO desc_cache_new (ticker, blob, ts) VALUES (?, ?, ?)",
                           (ticker, json.dumps(blob_dict), ts))
            cur.execute("DROP TABLE desc_cache")
            cur.execute("ALTER TABLE desc_cache_new RENAME TO desc_cache")
            conn.commit()
        elif 'blob' not in columns:
            # Table exists but missing blob column, add it
            cur.execute("ALTER TABLE desc_cache ADD COLUMN blob TEXT")
            conn.commit()
    else:
        # Create new table with correct schema
        cur.execute("""
            CREATE TABLE desc_cache (
                ticker TEXT PRIMARY KEY,
                blob TEXT,
                ts TEXT
            )
        """)
        conn.commit()
    return conn

def _cache_get(conn, ticker):
    cur = conn.cursor()
    cur.execute("SELECT blob, ts FROM desc_cache WHERE ticker = ?", (ticker.upper(),))
    row = cur.fetchone()
    if not row or row[0] is None:  # No row or NULL blob
        return None
    blob, ts = row
    if not blob or not ts:  # Empty blob or timestamp
        return None
    try:
        ts_dt = datetime.fromisoformat(ts)
    except Exception:
        return None
    if datetime.utcnow() - ts_dt > pd.Timedelta(hours=CACHE_TTL_HOURS):
        return None  # stale
    try:
        return json.loads(blob)
    except Exception:
        return None

def _cache_put(conn, ticker, blob_dict):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO desc_cache (ticker, blob, ts) VALUES (?, ?, ?)",
        (ticker.upper(), json.dumps(blob_dict), datetime.utcnow().isoformat())
    )
    conn.commit()

def fetch_metadata_live(ticker, retries=3, base_wait=1.5):
    """
    Fetch metadata from yfinance. Returns empty dict on 404 (not found).
    Caches 404s so we don't retry them.
    """
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            try:
                info = t.get_info()
            except AttributeError:
                info = t.info
            
            # Check if ticker not found (404 equivalent)
            if not info or info.get('symbol') is None:
                # Ticker not found - return empty and mark as 404
                return {"summary": "", "sector": "", "industry": "", "country": "", "website": "", "exchange": "", "_not_found": True}
            
            # yfinance exchange can be in 'exchange' field
            exchange_raw = info.get("exchange", "") or ""
            # Map common yfinance exchange codes to readable names
            exchange_map = {
                "NMS": "NASDAQ",  # NASDAQ Stock Market
                "NGM": "NASDAQ",  # NASDAQ Capital Market
                "NCM": "NASDAQ",  # NASDAQ Capital Market (alternate)
                "NYQ": "NYSE",    # New York Stock Exchange
                "ASE": "NYSE",    # NYSE American
                "PCX": "NYSE",    # NYSE Arca (archived)
                "BTS": "NYSE",    # NYSE Arca
            }
            exchange = exchange_map.get(exchange_raw, exchange_raw)
            # If still empty, try to derive from symbol suffixes
            if not exchange:
                symbol = info.get("symbol", "")
                if ".TO" in symbol or ".V" in symbol:
                    exchange = "TSX"
                elif ".L" in symbol:
                    exchange = "LSE"
                elif ".HK" in symbol:
                    exchange = "HKEX"
            return {
                "summary": info.get("longBusinessSummary", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "website": info.get("website", ""),
                "exchange": exchange,
                "_not_found": False
            }
        except Exception as e:
            msg = str(e).lower()
            # Check for 404/Not Found errors - don't retry these
            if "404" in msg or "not found" in msg or "quote not found" in msg:
                # Ticker doesn't exist - cache as not found
                return {"summary": "", "sector": "", "industry": "", "country": "", "website": "", "exchange": "", "_not_found": True}
            
            # Rate limiting - retry with backoff
            if "429" in msg or "too many requests" in msg:
                wait = base_wait * (2 ** attempt)
                time.sleep(wait)
                continue
            
            # Server errors - retry with backoff
            if any(code in msg for code in ("500", "502", "503", "504")):
                wait = base_wait * (2 ** attempt)
                time.sleep(wait)
                continue
            
            # Other errors on last attempt - return empty
            if attempt == retries - 1:
                return {"summary": "", "sector": "", "industry": "", "country": "", "website": "", "exchange": "", "_not_found": False}
    
    # All retries exhausted
    return {"summary": "", "sector": "", "industry": "", "country": "", "website": "", "exchange": "", "_not_found": False}

def load_existing_csv():
    """Load existing CSV if it exists, return dict of ticker -> row data."""
    if os.path.exists(OUT_PATH):
        try:
            df = pd.read_csv(OUT_PATH)
            # Fill NaN values with empty strings
            df = df.fillna('')
            existing = {}
            for _, row in df.iterrows():
                # Get ticker and convert to string, handle NaN/empty
                ticker_raw = row.get('ticker', '')
                if not ticker_raw or pd.isna(ticker_raw):
                    continue  # Skip rows with missing ticker
                ticker = str(ticker_raw).strip().upper()
                if ticker and ticker not in ('NAN', 'NONE', ''):
                    existing[ticker] = {
                        'summary': str(row.get('summary', '') or ''),
                        'sector': str(row.get('sector', '') or ''),
                        'industry': str(row.get('industry', '') or ''),
                        'country': str(row.get('country', '') or ''),
                        'website': str(row.get('website', '') or ''),
                        'exchange': str(row.get('exchange', '') or '')
                    }
            print(f"  Loaded {len(existing)} existing records from CSV")
            return existing
        except Exception as e:
            print(f"  Warning: Could not load existing CSV: {e}")
            import traceback
            traceback.print_exc()
    return {}


def get_metadata_df(tickers):
    conn = _init_cache()
    existing_csv = load_existing_csv()
    rows = []
    csv_count = 0
    cached_count = 0
    fetched_count = 0
    not_found_count = 0
    
    for tk in tickers:
        tk_upper = tk.upper()
        meta = None
        from_csv = False
        
        # Priority 1: Check existing CSV
        if tk_upper in existing_csv:
            csv_data = existing_csv[tk_upper]
            # Check if CSV has complete data
            if csv_data.get('summary') and csv_data.get('sector') and csv_data.get('industry'):
                # Use CSV data - no need to fetch
                meta = csv_data.copy()
                from_csv = True
                csv_count += 1
            else:
                # CSV has incomplete data - check cache/fetch
                meta = _cache_get(conn, tk)
        else:
            # Not in CSV - check cache
            meta = _cache_get(conn, tk)
        
        # Check if we should fetch fresh
        should_fetch = False
        if meta is None:
            # Not in CSV or cache - fetch
            should_fetch = True
        elif meta.get('_not_found', False):
            # Already marked as not found - skip (don't retry 404s)
            should_fetch = False
            not_found_count += 1
            if not from_csv:
                cached_count += 1
        elif not meta.get('sector') or not meta.get('industry') or not meta.get('exchange'):
            # Incomplete data - fetch fresh
            should_fetch = True
        elif not from_csv:
            # Complete data from cache
            cached_count += 1
        
        if should_fetch:
            meta = fetch_metadata_live(tk)
            fetched_count += 1
            if meta.get('_not_found', False):
                not_found_count += 1
            # Always cache the result (including 404s) so we don't retry
            _cache_put(conn, tk, meta)
        
        # Remove internal flag and ensure all required columns exist
        meta_clean = {k: v for k, v in meta.items() if k != '_not_found'}
        
        # Ensure all required columns are present (fill missing with empty strings)
        required_meta_cols = ['summary', 'sector', 'industry', 'country', 'website', 'exchange']
        for col in required_meta_cols:
            if col not in meta_clean:
                meta_clean[col] = ''
        
        # Convert all values to strings to avoid type issues
        for col in required_meta_cols:
            if meta_clean.get(col) is None:
                meta_clean[col] = ''
            else:
                meta_clean[col] = str(meta_clean[col])
        
        rows.append({"ticker": tk, **meta_clean})
        time.sleep(0.5)
    
    conn.close()
    print(f"  Sources: {csv_count} from CSV, {cached_count} from cache, {fetched_count} fetched, {not_found_count} not found (skipped)")
    
    # Create DataFrame and ensure all columns exist
    df = pd.DataFrame(rows)
    required_meta_cols = ['ticker', 'summary', 'sector', 'industry', 'country', 'website', 'exchange']
    for col in required_meta_cols:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)
    
    return df

def fetch_sec_universe(url=SEC_URL):
    print(f"Fetching SEC tickers...")
    r = requests.get(url, headers={'User-Agent':'CompFinder/1.0 you@email.com'})
    raw = r.json()
    records = []
    for entry in raw.values():
        rec = {
            "ticker": entry["ticker"],
            "cik": str(entry["cik_str"]).zfill(10),
            "name": entry["title"],
            "exchange": entry.get("exchange", "")
        }
        records.append(rec)
    return pd.DataFrame(records)

def main(limit=None, replace=False):
    """
    Build or update universe CSV.
    
    Args:
        limit: If set, process only this many tickers (for testing/incremental updates)
        replace: If True and limit is set, replace entire CSV. If False, merge with existing.
    """
    # Load existing CSV if it exists and we're not replacing
    existing_df = None
    if not replace and os.path.exists(OUT_PATH):
        try:
            existing_df = pd.read_csv(OUT_PATH)
            existing_df = existing_df.fillna('')
            print(f"  Found existing CSV with {len(existing_df)} records")
        except Exception as e:
            print(f"  Warning: Could not load existing CSV: {e}")
            existing_df = None
    
    # Fetch SEC tickers
    df_sec = fetch_sec_universe()
    
    # If limit is set and we're not replacing, check which tickers are new/missing
    if limit and not replace and existing_df is not None:
        existing_tickers = set(existing_df['ticker'].str.upper())
        # Only process tickers that are new or missing complete data
        new_tickers = []
        for ticker in df_sec['ticker'].head(limit * 2):  # Check more to find enough new ones
            if ticker.upper() not in existing_tickers:
                new_tickers.append(ticker)
            if len(new_tickers) >= limit:
                break
        if new_tickers:
            df_sec = df_sec[df_sec['ticker'].isin(new_tickers)]
            print(f"  Processing {len(new_tickers)} new tickers (out of {limit} requested)")
        else:
            print(f"  All requested tickers already exist in CSV")
            # Still process a few to update them if needed
            df_sec = df_sec.head(limit)
    elif limit:
        # Replace mode or no existing CSV - just process limit
        df_sec = df_sec.head(limit)
    
    # Process tickers - ensure SEC data is preserved
    df_sec = df_sec.rename(columns={'exchange': 'exchange_sec'})
    
    # Get metadata from yfinance (or CSV/cache)
    meta_df = get_metadata_df(df_sec['ticker'].tolist())
    
    # Merge: SEC data (ticker, cik, name, exchange_sec) + yfinance data (summary, sector, industry, country, website, exchange)
    df_new = pd.merge(df_sec, meta_df, on='ticker', how='left')
    
    # Ensure SEC columns (cik, name) are preserved and filled
    if 'cik' not in df_new.columns:
        df_new['cik'] = ''
    if 'name' not in df_new.columns:
        df_new['name'] = ''
    
    # Fill NaN in SEC columns with empty strings
    df_new['cik'] = df_new['cik'].fillna('').astype(str)
    df_new['name'] = df_new['name'].fillna('').astype(str)
    df_new['exchange_sec'] = df_new['exchange_sec'].fillna('').astype(str)
    
    # Handle exchange: prefer yfinance, fallback to SEC
    if 'exchange' not in df_new.columns:
        df_new['exchange'] = ''
    df_new['exchange'] = df_new['exchange'].fillna('').astype(str)
    
    # Set exchange: use yfinance exchange if available, otherwise use SEC exchange
    df_new['exchange'] = df_new.apply(
        lambda row: (
            str(row.get('exchange', '') or '').strip() or 
            str(row.get('exchange_sec', '') or '').strip() or 
            ''
        ),
        axis=1
    )
    
    # Drop the temporary exchange_sec column
    df_new = df_new.drop(columns=['exchange_sec'], errors='ignore')
    
    # Ensure all yfinance columns exist (fill missing with empty strings)
    yfinance_cols = ['sector', 'industry', 'country', 'summary', 'website']
    for col in yfinance_cols:
        if col not in df_new.columns:
            df_new[col] = ''
        df_new[col] = df_new[col].fillna('').astype(str)
    
    # Add source and updated_at
    df_new['source'] = 'yfinance'
    df_new['updated_at'] = datetime.utcnow().date().isoformat()
    
    # Final column order
    col_order = ['ticker','cik','name','exchange','sector','industry','country','summary','website','source','updated_at']
    
    # Ensure all columns exist and are filled
    for col in col_order:
        if col not in df_new.columns:
            df_new[col] = ''
        df_new[col] = df_new[col].fillna('').astype(str)
    
    # Select only required columns in correct order
    df_new = df_new[col_order]
    
    # Verify: all columns should have values (at least empty strings)
    print(f"  Processed {len(df_new)} tickers")
    missing_data = {}
    for col in col_order:
        empty_count = (df_new[col] == '').sum() + df_new[col].isna().sum()
        if empty_count > 0:
            missing_data[col] = empty_count
    if missing_data:
        print(f"  Warning: Empty values found: {missing_data}")
    else:
        print(f"  ✓ All columns filled for all tickers")
    
    # Merge with existing data
    if existing_df is not None and not replace:
        # Ensure existing_df has all required columns (fill missing with empty strings)
        required_cols = ['ticker','cik','name','exchange','sector','industry','country','summary','website','source','updated_at']
        for col in required_cols:
            if col not in existing_df.columns:
                existing_df[col] = ''
        
        # Remove existing records for tickers we just updated
        existing_df_clean = existing_df[~existing_df['ticker'].str.upper().isin(df_new['ticker'].str.upper())]
        
        # Ensure both DataFrames have the same columns in the same order
        existing_df_clean = existing_df_clean[col_order]
        df_new = df_new[col_order]
        
        # Fill NaN values with empty strings before concatenating
        existing_df_clean = existing_df_clean.fillna('')
        df_new = df_new.fillna('')
        
        # Combine: existing (without updated) + new/updated
        df_final = pd.concat([existing_df_clean, df_new], ignore_index=True)
        print(f"  Merged: {len(existing_df_clean)} existing + {len(df_new)} new/updated = {len(df_final)} total")
    else:
        # No existing data or replace mode - use only new data
        df_final = df_new.fillna('')
        print(f"  Writing {len(df_final)} records (replace mode)")
    
    # Ensure all required columns exist and are filled
    required_cols = ['ticker','cik','name','exchange','sector','industry','country','summary','website','source','updated_at']
    for col in required_cols:
        if col not in df_final.columns:
            df_final[col] = ''
        # Fill any remaining NaN values with empty strings
        df_final[col] = df_final[col].fillna('')
    
    # Ensure column order
    df_final = df_final[required_cols]
    
    # Deduplicate by CIK (keep one ticker per CIK)
    print(f"\n  Deduplicating by CIK...")
    print(f"    Before: {len(df_final):,} records, {df_final['cik'].nunique():,} unique CIKs")
    
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
    
    df_final = df_final.groupby('cik', group_keys=False).apply(select_primary_ticker).reset_index(drop=True)
    print(f"    After: {len(df_final):,} records, {df_final['cik'].nunique():,} unique CIKs")
    
    # Sort by ticker for consistency
    df_final = df_final.sort_values('ticker').reset_index(drop=True)
    
    # Write to CSV
    df_final.to_csv(OUT_PATH, index=False)
    print(f"✓ Wrote {OUT_PATH} ({len(df_final)} records)")
    print(f"  Columns: {', '.join(df_final.columns)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers to process (merges with existing CSV by default).")
    parser.add_argument("--replace", action='store_true', help="Replace entire CSV instead of merging (use with --limit for testing).")
    args = parser.parse_args()
    main(limit=args.limit, replace=args.replace)
