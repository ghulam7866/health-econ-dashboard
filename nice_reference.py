"""
nice_reference.py
------------------
Generates the curated reference table for NICE cost-effectiveness thresholds.

This script creates a reference table of NICE QALY threshold policy events
over time, which is used for annotation in the dashboard.

Usage:
    python nice_reference.py

Output:
    data/processed/nice_clean.csv

Last updated: 2026-07-02
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ROWS = [
    {
        "effective_date": "1999-04-01",
        "category": "Standard (Inferred)",
        "lower_gbp_per_qaly": 20000,
        "upper_gbp_per_qaly": 30000,
        "description": "NICE established. Informal 20k-30k range inferred from decisions."
    },
    {
        "effective_date": "2004-01-01",
        "category": "Standard Formalised",
        "lower_gbp_per_qaly": 20000,
        "upper_gbp_per_qaly": 30000,
        "description": "Formal state of 20k-30k standard range appraisal standard."
    },
    {
        "effective_date": "2009-01-01",
        "category": "End of Life Premium",
        "lower_gbp_per_qaly": 20000,
        "upper_gbp_per_qaly": 50000,
        "description": "Acceptable range raised to 50k for terminal conditions."
    },
    {
        "effective_date": "2013-01-01",
        "category": "HST Launch",
        "lower_gbp_per_qaly": None,
        "upper_gbp_per_qaly": None,
        "description": "Highly Specialised Technologies programme launched."
    },
    {
        "effective_date": "2017-01-01",
        "category": "HST Threshold Set",
        "lower_gbp_per_qaly": 100000,
        "upper_gbp_per_qaly": 300000,
        "description": "HST standard set to 100k, scaling with weighting up to 300k."
    },
    {
        "effective_date": "2022-01-01",
        "category": "Severity Modifier",
        "lower_gbp_per_qaly": 20000,
        "upper_gbp_per_qaly": 36000,
        "description": "Severity modifier replaces EOL, raising standard max toward 36k."
    },
    {
        "effective_date": "2025-04-01",
        "category": "HST Reconfirmed",
        "lower_gbp_per_qaly": 100000,
        "upper_gbp_per_qaly": 300000,
        "description": "HST extensions reconfirmed at 300k upper tier max."
    },
    {
        "effective_date": "2025-12-01",
        "category": "Policy Announcement",
        "lower_gbp_per_qaly": 25000,
        "upper_gbp_per_qaly": 35000,
        "description": "UK gov confirms standard threshold increase ahead of April 2026 launch."
    },
    {
        "effective_date": "2026-04-01",
        "category": "New Standard Active",
        "lower_gbp_per_qaly": 25000,
        "upper_gbp_per_qaly": 35000,
        "description": "New standard threshold baseline (25,000-35,000/QALY) goes live."
    }
]


def generate():
    """Generate the NICE reference table and save to CSV."""
    df = pd.DataFrame(ROWS)
    df["date"] = pd.to_datetime(df["effective_date"])
    df = df.sort_values("date").reset_index(drop=True)

    out_path = PROCESSED_DIR / "nice_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Production table written ({len(df)} lines) -> {out_path}")


if __name__ == "__main__":
    generate()