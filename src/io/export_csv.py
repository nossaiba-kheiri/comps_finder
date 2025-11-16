"""
export_csv.py: Export DataFrames/results to output CSV files for KNN and ranked outputs.
"""
import os
import pandas as pd


def export_leaderboard(df, path, leaderboard_type='ranked'):
    """
    Export leaderboard to CSV.
    leaderboard_type: 'knn' or 'ranked'
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Ensure required columns exist
    if leaderboard_type == 'knn':
        required_cols = ['rank_knn', 'knn_score', 'ticker', 'name', 'exchange']
        for col in required_cols:
            if col not in df.columns:
                if col == 'rank_knn':
                    df['rank_knn'] = range(1, len(df) + 1)
                elif col == 'knn_score' and 'S_fast' in df.columns:
                    df['knn_score'] = df['S_fast']
    
    elif leaderboard_type == 'ranked':
        required_cols = ['rank_ml', 'ml_score', 'ticker', 'name', 'exchange', 'P', 'C', 'M', 'S', 'I', 'E', 'R']
        for col in required_cols:
            if col not in df.columns:
                if col == 'rank_ml':
                    df['rank_ml'] = range(1, len(df) + 1)
                elif col == 'ml_score' and 'score' in df.columns:
                    df['ml_score'] = df['score']
    
    # Save to CSV
    df.to_csv(path, index=False)
    print(f"✓ Exported {leaderboard_type} leaderboard to: {path} ({len(df)} rows)")


if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'name': ['Apple', 'Microsoft'],
        'score': [95.0, 90.0]
    })
    export_leaderboard(df, '/tmp/test_ranked.csv', 'ranked')
