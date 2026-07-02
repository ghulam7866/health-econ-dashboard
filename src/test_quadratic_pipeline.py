"""
test_quadratic_pipeline.py
---------------------------
Tests that the quadratic trend flows correctly through the pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"


def test_quadratic_trend():
    print("=" * 70)
    print("TESTING QUADRATIC TREND PIPELINE")
    print("=" * 70)
    
    # Load combined data
    if not COMBINED_PATH.exists():
        print("❌ ERROR: combined_quarterly.csv not found!")
        print("   Please run the pipeline first: python run_pipeline.py")
        return False
    
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])
    
    print(f"\n✅ Combined data loaded: {len(df)} rows")
    
    # Check for quadratic_trend
    if "quadratic_trend" not in df.columns:
        print("❌ ERROR: quadratic_trend column missing from combined data!")
        print(f"   Available columns: {df.columns.tolist()}")
        return False
    
    print("✅ quadratic_trend column found in combined data")
    
    # Check RTT specifically
    rtt_data = df[df["metric"] == "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data"]
    
    if rtt_data.empty:
        print("❌ ERROR: RTT data not found!")
        return False
    
    print(f"✅ RTT data found: {len(rtt_data)} rows")
    
    # Check if quadratic_trend has values
    if rtt_data["quadratic_trend"].isna().all():
        print("❌ ERROR: quadratic_trend is all NaN for RTT!")
        return False
    
    print(f"✅ quadratic_trend has values (range: {rtt_data['quadratic_trend'].min():.3f} to {rtt_data['quadratic_trend'].max():.3f})")
    
    # Check exog_config activation
    try:
        from exog_config import EXOG_CONFIG
        if "quadratic_trend" not in EXOG_CONFIG.get("RTT waiting list (level)", []):
            print("⚠ WARNING: quadratic_trend not in EXOG_CONFIG for RTT!")
            print("   Please uncomment it in exog_config.py")
        else:
            print("✅ quadratic_trend configured in EXOG_CONFIG")
    except ImportError:
        print("⚠ WARNING: Could not import exog_config to verify configuration")
    
    # Check that RTT has the variable in the data
    print("\n" + "=" * 70)
    print("SAMPLE RTT DATA WITH QUADRATIC TREND")
    print("=" * 70)
    print(rtt_data[["quarter", "value", "quadratic_trend"]].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    return True


if __name__ == "__main__":
    test_quadratic_trend()