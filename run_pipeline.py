"""
run_pipeline.py
----------------
Master end-to-end execution script for the Health Economics Dashboard.

This script sequentially executes all pipeline stages:
1. NICE reference table generation
2. Raw data scraping
3. Data cleaning
4. Spot check validation
5. Quarterly alignment and merge
6. Exogenous dummy injection
7. SARIMAX forecasting

Usage:
    python run_pipeline.py

Last updated: 2026-07-02
"""

import sys
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"

PIPELINE_STACK = [
    ("NICE Reference Table Generation", ROOT_DIR / "nice_reference.py"),
    ("Raw Data Scraper Smoke Test", ROOT_DIR / "test_scraper.py"),
    ("Tidy Extraction & Cleaner Engine", SRC_DIR / "cleaner.py"),
    ("Pre-Aggregation Spot Check Validation", ROOT_DIR / "spot_check.py"),
    ("Quarterly Resampling & Alignment Merge", SRC_DIR / "align_merge.py"),
    ("Exogenous Dummy Interventions Injector", SRC_DIR / "add_intervention_dummies.py"),
    ("Master SARIMAX 6-Year Forecasting Engine", SRC_DIR / "master_forecast_engine.py")
]


def run_step(name: str, script_path: Path) -> bool:
    """
    Execute a single pipeline step.

    Parameters
    ----------
    name : str
        Name of the step for logging
    script_path : Path
        Path to the script to execute

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print(f"\n[STEP] Starting: {name}")
    print(f"       Executing: {script_path.relative_to(ROOT_DIR) if ROOT_DIR in script_path.parents else script_path.name}")
    print("-" * 60)

    if not script_path.exists():
        print(f"[ERROR] Script file target does not exist: {script_path}")
        return False

    start_time = time.time()
    process = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT_DIR))
    duration = time.time() - start_time

    if process.returncode == 0:
        print(f"[SUCCESS] {name} completed cleanly in {duration:.2f}s")
        return True
    else:
        print(f"[FAILURE] {name} exited with non-zero code {process.returncode}")
        return False


def main():
    """Main entry point - runs the full pipeline."""
    print("=" * 75)
    print("HEALTH ECON DASHBOARD DATA PIPELINE RUNNER")
    print("=" * 75)

    pipeline_start = time.time()

    for name, path in PIPELINE_STACK:
        success = run_step(name, path)
        if not success:
            print("\n[CRITICAL] PIPELINE TERMINATED: Step execution crash encountered.")
            print("Please fix the stack trace error highlighted above before moving on.")
            sys.exit(1)

    total_time = time.time() - pipeline_start

    print("\n" + "=" * 75)
    print("FULL DATA PIPELINE REBUILT SUCCESSFULLY")
    print(f"Total Execution Horizon: {total_time:.2f} seconds")
    print("Target Deliverable Loaded: data/processed/forecast_6yr.csv")
    print("=" * 75)


if __name__ == "__main__":
    main()