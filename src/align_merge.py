"""
align_merge.py
---------------
Filters each processed CSV down to the confirmed headline metrics, then
resamples everything onto a shared quarterly calendar so they can be
combined into one dataset for the dashboard.

Frequency choice: quarterly. Beds (KH03) is already quarterly and is the
coarsest genuinely-periodic source, so monthly sources are downsampled to
match rather than upsampling Beds to monthly (which would just be
interpolation, not real data).

Aggregation rule depends on what each metric actually represents:
  - FLOW  (events during the period, e.g. appointments, completed
           pathways, attendances)        -> summed across the quarter
  - STOCK (snapshot at a point in time, e.g. number currently waiting)
                                          -> averaged across the quarter
  - RATE  (already a ratio/percentage/median, e.g. % within 18 weeks)
                                          -> averaged across the quarter
  - Workforce FTE (a level reported as of a date)
                                          -> last month's value in the quarter

Annual sources (Population, PESA) are forward-filled across the quarters
of their year, since no finer-grained truth exists. This is a real
methodological approximation - flag it in your write-up rather than
treating it as genuinely quarterly data.

NICE is NOT merged in here - it's 9 discrete policy events, not a time
series. Use nice_clean.csv separately as annotation markers (e.g.
vertical lines at policy dates) when plotting, not as a merged column.

UPDATED: 2026-07-01 - Added quadratic_trend for RTT waiting list curvature.

Run:
    python align_merge.py

Output:
    data/processed/combined_quarterly.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
OUT_PATH = PROCESSED_DIR / "combined_quarterly.csv"

# ---------------------------------------------------------------------------
# Headline metrics, each mapped to its aggregation rule for monthly -> quarterly
# ---------------------------------------------------------------------------

RTT_METRICS = {
    "Incomplete RTT pathways - % within 18 weeks": "mean",
    "Incomplete RTT pathways - Median wait (weeks)": "mean",
    "Incomplete RTT pathways - Total waiting (mil) with estimates for missing data": "mean",
    "Incomplete RTT pathways - No. within 18 weeks with estimates for missing data": "mean",
    "Incomplete RTT pathways - No. > 18 weeks with estimates for missing data": "mean",
    "Incomplete RTT pathways - No. > 52 weeks with estimates for missing data": "mean",
    "Incomplete RTT pathways - No. > 104 weeks with estimates for missing data": "mean",
    "New RTT periods - No. of new RTT periods": "sum",
    "Total removals - Total removals with estimates": "sum",
    "Total completed pathways - Total completed pathways with estimates": "sum",
    "Completed non-admitted RTT pathways - No. within 18 weeks": "sum",
    "Completed non-admitted RTT pathways - No. > 52 weeks": "sum",
    "Completed non-admitted RTT pathways - Median wait (weeks)": "mean",
    "Completed admitted (unadjusted) RTT pathways - No. of pathways (all) with estimates for missing data": "sum",
    "Completed admitted (unadjusted) RTT pathways - No. within 18 weeks": "sum",
    "Completed admitted (unadjusted) RTT pathways - Median wait (weeks)": "mean",
    "Unique Patients - Ratio of total unique patients to total waiting (i.e. incomplete RTT pathways)": "mean",
}

AE_METRICS = {
    "total_attendances": "sum",
    "type_1_departments_major_ae": "sum",
    "type_3_departments_other_ae_minor_injury_unit": "sum",
    "total_emergency_admissions_via_ae": "sum",
    "emergency_admissions_via_type_1_ae": "sum",
    "emergency_admissions_via_type_2_ae": "sum",
    "emergency_admissions_via_type_3_and_4_ae": "sum",
    "number_of_patients_spending_12_hours_from_decision_to_admit_to_admission": "sum",
}

GP_METRICS = {
    "total_attended_appointments": "sum",
    "attended_face_to_face": "sum",
    "attended_telephone": "sum",
    "attended_video_conference_online": "sum",
    "attended_home_visit": "sum",
}

WORKFORCE_METRICS = {
    "FTE: All staff groups - All staff groups": "last",
    "FTE: Professionally qualified clinical staff - All staff groups": "last",
    "FTE: Professionally qualified clinical staff - Nurses & health visitors": "last",
    "FTE: Support to clinical staff - Support to doctors, nurses & midwives": "last",
    "FTE: Professionally qualified clinical staff - HCHS doctors - All grades": "last",
    "FTE: NHS infrastructure support - Central functions": "last",
    "FTE: Professionally qualified clinical staff - HCHS doctors - Consultant": "last",
    "FTE: Professionally qualified clinical staff - HCHS doctors - Specialty Registrar": "last",
    "FTE: Professionally qualified clinical staff - HCHS doctors - Core Training": "last",
    "FTE: Professionally qualified clinical staff - Midwives": "last",
    "FTE: Professionally qualified clinical staff - HCHS doctors - Foundation Doctor Year 1": "last",
    "FTE: Professionally qualified clinical staff - HCHS doctors - Foundation Doctor Year 2": "last",
    "FTE: Professionally qualified clinical staff - HCHS doctors - Other HCHS Doctor Grades": "last",
}

BEDS_METRIC = "total_occupied_beds_overnight"


def _quarter_label(date: pd.Series) -> pd.Series:
    """Map any date onto its calendar-quarter start date (e.g. a date in
    May 2024 -> 2024-04-01). Used as the common key across all sources."""
    return date.dt.to_period("Q").dt.to_timestamp()


def resample_monthly_source(path: Path, metric_aggs: dict, source_label: str) -> pd.DataFrame:
    """Filter to the given metrics, then resample monthly -> quarterly using
    a per-metric aggregation rule (sum / mean / last)."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["metric"].isin(metric_aggs.keys())].copy()
    df["quarter"] = _quarter_label(df["date"])

    frames = []
    for metric, agg in metric_aggs.items():
        sub = df[df["metric"] == metric]
        if sub.empty:
            print(f"  ⚠ {source_label}: metric not found in file - '{metric}'")
            continue
        grouped = sub.groupby("quarter")["value"].agg(agg).reset_index()
        grouped["metric"] = metric
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=["quarter", "value", "metric"])
    return pd.concat(frames, ignore_index=True)


def resample_beds(path: Path) -> pd.DataFrame:
    """Beds is already quarterly - just align snapshot dates onto quarter
    start labels so they line up with the resampled sources."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quarter"] = _quarter_label(df["date"])
    df["metric"] = BEDS_METRIC
    return df[["quarter", "value", "metric"]]


def expand_annual_to_quarters(path: Path, value_col: str, metric_name: str,
                               quarters_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill an annual series across the quarters of its year. This
    is an approximation - flag it in your methodology write-up."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.set_index("date")[[value_col]].rename(columns={value_col: "value"}).sort_index()

    full = df.reindex(df.index.union(quarters_index)).sort_index()
    full["value"] = full["value"].ffill()
    full = full.loc[full.index.isin(quarters_index)].reset_index().rename(columns={"index": "quarter"})
    full["metric"] = metric_name
    return full[["quarter", "value", "metric"]]


def expand_pesa_to_quarters(path: Path, quarters_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Same forward-fill approach as population, applied per metric since
    PESA keeps its full, unfiltered COFOG breakdown."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    frames = []
    for metric, sub in df.groupby("metric"):
        sub = sub.set_index("date")[["value"]].sort_index()
        full = sub.reindex(sub.index.union(quarters_index)).sort_index()
        full["value"] = full["value"].ffill()
        full = full.loc[full.index.isin(quarters_index)].reset_index().rename(columns={"index": "quarter"})
        full["metric"] = metric
        frames.append(full[["quarter", "value", "metric"]])
    return pd.concat(frames, ignore_index=True)


def add_quadratic_trend(df, date_col='quarter'):
    """
    Add quadratic trend variable for series with significant curvature.
    Centered and scaled to reduce multicollinearity.
    
    Parameters
    ----------
    df : DataFrame
        Must have a date column (default: 'quarter')
    date_col : str
        Name of the date column
        
    Returns
    -------
    DataFrame with 'quadratic_trend' column added
    """
    # Get unique quarters sorted
    unique_quarters = sorted(df[date_col].unique())
    n = len(unique_quarters)
    
    # Create time index (0 to n-1) for the full timeline
    t = np.arange(n)
    
    # Center the time variable to reduce multicollinearity
    t_centered = t - np.mean(t)
    
    # Quadratic trend (centered and scaled to unit variance)
    quadratic = (t_centered ** 2)
    quadratic_scaled = quadratic / np.std(quadratic) if np.std(quadratic) > 0 else quadratic
    
    # Create mapping from quarter to quadratic trend value
    quarter_to_quadratic = {q: val for q, val in zip(unique_quarters, quadratic_scaled)}
    
    # Add column to dataframe
    df['quadratic_trend'] = df[date_col].map(quarter_to_quadratic)
    
    print(f"  ✓ Added quadratic_trend column ({n} quarters, range: {df['quadratic_trend'].min():.3f} → {df['quadratic_trend'].max():.3f})")
    
    return df


def main():
    print("Aligning all sources to quarterly frequency...\n")

    rtt = resample_monthly_source(PROCESSED_DIR / "rtt_clean.csv", RTT_METRICS, "RTT")
    rtt["source"] = "RTT"

    ae = resample_monthly_source(PROCESSED_DIR / "ae_clean.csv", AE_METRICS, "A&E")
    ae["source"] = "A&E"

    gp = resample_monthly_source(PROCESSED_DIR / "gp_appointments_clean.csv", GP_METRICS, "GP")
    gp["source"] = "GP Appointments"

    workforce = resample_monthly_source(PROCESSED_DIR / "workforce_clean.csv", WORKFORCE_METRICS, "Workforce")
    workforce["source"] = "Workforce"

    beds = resample_beds(PROCESSED_DIR / "beds_clean.csv")
    beds["source"] = "Beds"

    # Master quarterly calendar = union of every quarter present in the
    # monthly/quarterly sources, so annual sources can be forward-filled
    # onto all of them.
    all_quarters = pd.concat([rtt["quarter"], ae["quarter"], gp["quarter"], workforce["quarter"], beds["quarter"]])
    quarters_index = pd.DatetimeIndex(sorted(all_quarters.dropna().unique()))

    population = expand_annual_to_quarters(
        PROCESSED_DIR / "population_clean.csv", "population", "uk_population", quarters_index
    )
    population["source"] = "ONS Population"

    pesa = expand_pesa_to_quarters(PROCESSED_DIR / "pesa_clean.csv", quarters_index)
    pesa["source"] = "PESA"

    combined = pd.concat([rtt, ae, gp, workforce, beds, population, pesa], ignore_index=True)
    combined = combined.sort_values(["source", "metric", "quarter"]).reset_index(drop=True)

    print(f"\nCombined data: {len(combined)} rows, {combined['metric'].nunique()} metrics")
    print(f"Quarter range: {combined['quarter'].min()} → {combined['quarter'].max()}")

    # ADD: Add quadratic_trend (and placeholder COVID dummies)
    # This MUST be done BEFORE saving
    print("\nAdding exogenous variables...")
    combined = add_quadratic_trend(combined, 'quarter')
    
    # Add placeholder columns for COVID dummies (they'll be overwritten by add_intervention_dummies.py)
    # But we need them to exist in the CSV
    combined['covid_pulse'] = 0
    combined['post_covid_regime'] = 0
    combined['post_covid_trend_break'] = 0

    # Verify quadratic_trend was added
    if 'quadratic_trend' in combined.columns:
        print(f"  ✓ quadratic_trend column present (range: {combined['quadratic_trend'].min():.3f} → {combined['quadratic_trend'].max():.3f})")
    else:
        print("  ❌ ERROR: quadratic_trend column NOT added!")
        return

    combined.to_csv(OUT_PATH, index=False)
    print(f"\n✓ Saved combined quarterly dataset → {OUT_PATH}")
    print(f"  {len(combined)} rows, {combined['metric'].nunique()} metrics")
    print(f"  Columns: {combined.columns.tolist()}")

    # Flag sources with much shorter coverage than the overall range
    print("\nCoverage per source:")
    for src, sub in combined.groupby("source"):
        print(f"  {src:<18} {sub['quarter'].min()} → {sub['quarter'].max()}")

    print("\nNote: NICE QALY thresholds were NOT merged in - they're 9 discrete")
    print("policy events, not a time series. Use nice_clean.csv separately as")
    print("annotation markers on charts (e.g. vertical lines at policy dates).")


if __name__ == "__main__":
    main()