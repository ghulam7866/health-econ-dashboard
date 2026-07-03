"""
test_scraper_beds_gap.py
--------------------------
Downloads the 8 missing quarterly KH03 overnight beds files (Q1 2024-25
through Q4 2025-26) to close the gap in the existing bulk CSV, which
stopped at 2024-04-01.

These are individual quarterly "Web_File" releases, likely a different
layout to the bulk CSV - inspect structure after downloading, before
extending the cleaner.

Run:
    python test_scraper_beds_gap.py
"""

import requests
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.england.nhs.uk/",
}

BEDS_GAP_URLS = [
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2024/11/Beds-Open-Overnight-Web_File-Q1-2024-25-revised.xlsx",
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2024/11/Beds-Open-Overnight-Web_File-Q2-2024-25.xlsx",
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/11/Beds-Open-Overnight-Web_File-Q3-2024-25-revised.xlsx",
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/05/Beds-Open-Overnight-Web_File-Q4-2024-25.xlsx",
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/11/Beds-Open-Overnight-Web_File-Q1-2025-26-revised.xlsx",
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/11/Beds-Open-Overnight-Web_File-Q2-2025-26.xlsx",
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2026/02/Beds-Open-Overnight-Web_File-Q3-2025-26.xlsx",
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2026/05/Beds-Open-Overnight-Web_File-Q4-2025-26.xlsx",
]


def main():
    print("=" * 70)
    print("Downloading 8 quarterly KH03 'Web_File' releases (gap fill)")
    print("=" * 70)

    results = {}
    for url in BEDS_GAP_URLS:
        fname = Path(url).name
        dest = RAW_DIR / fname

        if dest.exists():
            print(f"  ✓ cached ({dest.stat().st_size/1024:.0f} KB) → {fname}")
            results[fname] = dest
            continue

        try:
            print(f"  ↓ {fname}...")
            with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "")
                if "text/html" in ct:
                    print(f"    ✗ Got HTML — URL may be wrong")
                    results[fname] = None
                    continue
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
            print(f"    ✓ {dest.stat().st_size/1024:.0f} KB saved")
            results[fname] = dest
        except requests.HTTPError as e:
            print(f"    ✗ HTTP {e.response.status_code}")
            results[fname] = None
        except Exception as e:
            print(f"    ✗ {e}")
            results[fname] = None

    print("\n" + "=" * 70)
    success = sum(1 for v in results.values() if v)
    print(f"Downloaded/cached: {success}/{len(results)}")
    if success < len(results):
        print("\nFailed files:")
        for fname, path in results.items():
            if not path:
                print(f"  • {fname}")


if __name__ == "__main__":
    main()
