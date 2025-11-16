"""
recency.py: Recency scoring (R), 1.0 if updated <=24mo, linear decay to 0 at 60mo.
"""
from datetime import datetime, timedelta


def score_recency(updated_at):
    """
    Compute recency score R.
    updated_at: ISO date string or datetime object
    Returns score in [0, 1].
    """
    if not updated_at:
        return 0.5  # Neutral if unknown
    
    # Parse date
    if isinstance(updated_at, str):
        try:
            updated_dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        except:
            try:
                updated_dt = datetime.strptime(updated_at, '%Y-%m-%d')
            except:
                return 0.5
    else:
        updated_dt = updated_at
    
    # Current date
    now = datetime.utcnow()
    if updated_dt.tzinfo:
        now = now.replace(tzinfo=updated_dt.tzinfo)
    
    # Compute months since update
    delta = now - updated_dt
    months = delta.days / 30.0
    
    # Score: 1.0 if <=24 months, linear decay to 0 at 60 months
    if months <= 24:
        return 1.0
    elif months >= 60:
        return 0.0
    else:
        # Linear decay
        return 1.0 - ((months - 24) / 36.0)


if __name__ == "__main__":
    # Test
    recent = datetime.utcnow() - timedelta(days=180)  # 6 months ago
    old = datetime.utcnow() - timedelta(days=900)  # 30 months ago
    very_old = datetime.utcnow() - timedelta(days=2000)  # 66 months ago
    
    print(f"Recent (6mo): {score_recency(recent)}")
    print(f"Old (30mo): {score_recency(old)}")
    print(f"Very old (66mo): {score_recency(very_old)}")
