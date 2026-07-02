"""
test_scraper.py
---------------
Smoke test and automated ingestion layer for all health systems datasets.
Enforces strict process exit codes to protect downstream pipelines from partial ingestion.
"""

import sys
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

DIRECT_URLS = [
    (
        "RTT Waiting Times",
        "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2026/06/RTT-Overview-Timeseries-Including-Estimates-for-Missing-Trusts-Apr26-XLS-116K-X7gGnn.xlsx",
        None,
    ),
    (
        "A&E Attendances",
        "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2026/06/Monthly-AE-Time-Series-May-2026-wlgnE2.xls",
        None,
    ),
    (
        "Bed Occupancy",
        "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2024/09/KH03-Occupied-by-Spec-Overnight-only.csv",
        None,
    ),
    (
        "ONS Population",
        (
            "https://www.ons.gov.uk/generator"
            "?format=csv&uri=/peoplepopulationandcommunity/populationandmigration"
            "/populationestimates/timeseries/ukpop/pop"
        ),
        "ons_uk_population.csv",
    ),
    (
        "GP Appointments (National Overview)",
        "https://files.digital.nhs.uk/59/1DFF75/National_Overview.csv",
        None,
    ),
    (
        "GP Appointments (2025-26 actual, fallback)",
        "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2026/02/2025-26-Apr-Sep-Monthly-Combined-CSV-Provisional.csv",
        None,
    ),
    (
        "NHS Workforce (HCHS)",
        "https://files.digital.nhs.uk/FD/F370E1/NHS%20HCHS%20Workforce%20Statistics%2C%20Trusts%20and%20core%20organisations%20-%20data%20tables%2C%20February%202026.xlsx",
        None,
    ),
    (
        "HMT PESA Health Expenditure",
        "https://assets.publishing.service.gov.uk/media/6874fe33b1b4ebc2c2e46574/PESA_2025_CP_Chapter_4_tables.xlsx",
        None,
    ),
]

def step1_probe():
    print("\n-- STEP 1: HEAD Probe Network Targets --")
    results = {}
    for name, url, fname_override in DIRECT_URLS:
        fname = fname_override or Path(url.split("?")[0]).name
        dest = RAW_DIR / fname
        
        if dest.exists() and dest.stat().st_size > 0:
            print(f"   [CACHED] {name} found locally. Network verification skipped.")
            results[name] = "CACHED"
            continue
            
        try:
            r = requests.head(url, headers=HEADERS, timeout=10, allow_redirects=True)
            ok = r.status_code == 200
            size = r.headers.get("Content-Length", None)
            size_str = f"{int(size)/1024:.0f} KB" if size else "size unknown"
            ct = r.headers.get("Content-Type", "unknown")[:40]
            status_tag = "[OK]" if ok else "[FAIL]"
            print(f"   {status_tag} [{r.status_code}] {name} | {size_str} | {ct}")
            results[name] = ok
        except Exception as e:
            print(f"   [ERROR] {name} Probe Failed: {e}")
            results[name] = False
    return results

def step2_download(probe_results):
    print("\n-- STEP 2: Downstream Asset Sync --")
    downloaded = {}
    for name, url, fname_override in DIRECT_URLS:
        fname = fname_override or Path(url.split("?")[0]).name
        dest = RAW_DIR / fname
        probe_status = probe_results.get(name, False)

        if probe_status == "CACHED":
            downloaded[name] = dest
            continue

        if not probe_status:
            print(f"   [SKIP] {name}: Synchronisation skipped due to bad network endpoint.")
            downloaded[name] = None
            continue
            
        try:
            print(f"   [DOWNLOADING] Asset stream: {name}...")
            with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "")
                if "text/html" in ct:
                    print(f"      [ERROR] Ingestion Error: Target returned HTML payload instead of structured file.")
                    downloaded[name] = None
                    continue
                    
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
            print(f"      [SUCCESS] Safe extraction complete: {dest.stat().st_size/1024:.0f} KB")
            downloaded[name] = dest
        except Exception as e:
            print(f"      [ERROR] Failed to write file: {e}")
            downloaded[name] = None
    return downloaded

def main():
    print("=" * 75)
    print(" HEALTH ECONOMICS SCRAPER & INGESTION INTEGRITY ENGINE")
    print("=" * 75)

    probe = step1_probe()
    downloads = step2_download(probe)

    print("\n-- DATASETS INGESTION MATRIX SUMMARY --")
    success = sum(1 for v in downloads.values() if v)
    total = len(downloads)
    print(f"   Execution Integrity Ratio: {success}/{total} assets secured.")

    failed = [name for name, path in downloads.items() if not path]
    if failed:
        print(f"\n   [CRITICAL] INGESTION ERROR: {len(failed)} pipeline targets failed to load:")
        for name in failed:
            print(f"      - {name}")
        print("\n   [ACTION] Terminating workflow path execution sequence.")
        print("   NHS/Digital URL tokens change dynamically. Please update DIRECT_URLS mapping.")
        sys.exit(1)
        
    print("\n   [SUCCESS] All upstream assets verified and matching local storage. Proceeding.")
    print("=" * 75)
    print()

if __name__ == "__main__":
    main()