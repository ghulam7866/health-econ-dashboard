"""
cleaner.py
----------
Cleans and standardises raw NHS / ONS / HMT / NICE data into tidy
DataFrames ready for forecasting and visualisation.

Each source has its own cleaning function that handles data‑specific quirks
(split headers, non‑standard date formats, etc.).  The output is a
consistent `date | metric | value | source` format written to
`data/processed/`.

Run:
    python src/cleaner.py
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\raw")
PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. RTT Waiting Times
#    Header is split across rows 10 (section title) and 11 (column names).
#    Dates are in column index 2 (col B is year, col C is the datetime).
#    Data starts at row 12.
# ---------------------------------------------------------------------------
def clean_rtt() -> pd.DataFrame:
    path = _find(RAW_DIR, "RTT-Overview")
    print(f"[RTT] Reading {path.name}...")

    # Read the two header rows to build combined column names.
    # The top row is a section label (e.g. 'Incomplete Pathways') that spans
    # several merged columns.  We forward‑fill it so that every bottom‑row
    # column gets a unique, section‑prefixed name.
    header_top = pd.read_excel(path, sheet_name=0, header=None,
                                skiprows=10, nrows=1).iloc[0].tolist()
    header_bot = pd.read_excel(path, sheet_name=0, header=None,
                                skiprows=11, nrows=1).iloc[0].tolist()

    top_filled = pd.Series(header_top).replace("nan", pd.NA).ffill()

    col_names = []
    for top, bot in zip(top_filled, header_bot):
        top = str(top).strip() if pd.notna(top) else ""
        bot = str(bot).strip() if str(bot) != "nan" else ""
        if top and bot:
            col_names.append(f"{top} - {bot}")
        elif bot:
            col_names.append(bot)
        elif top:
            col_names.append(top)
        else:
            col_names.append("drop")

    # Read actual data from row 12 onwards
    df = pd.read_excel(path, sheet_name=0, header=None, skiprows=12)
    df.columns = col_names[:len(df.columns)]
    print(f"[RTT] Columns: {col_names[:8]}")

    # Column index 2 (0‑based) is the date – it already came through as datetime
    df = df.rename(columns={df.columns[2]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Drop the leading blank column and the redundant year column
    df = df.drop(columns=[c for c in df.columns if c in ["drop", "Year"]], errors="ignore")

    # Replace '-' placeholders with NaN, coerce value columns to numeric
    df = df.replace("-", np.nan)
    for col in df.columns:
        if col != "date" and isinstance(df[col], pd.Series):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    tidy = (
        df[["date"] + num_cols]
        .melt(id_vars="date", var_name="metric", value_name="value")
        .dropna(subset=["value"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    tidy["source"] = "NHS England RTT"

    _save(tidy, "rtt_clean.csv")
    print(f"[RTT] Done — {len(tidy)} rows, {tidy['metric'].nunique()} metrics")
    return tidy


# ---------------------------------------------------------------------------
# 2. A&E Attendances
#    Header is row 13.  Date column is index 1 (col B).  Data starts row 14.
# ---------------------------------------------------------------------------
def clean_ae() -> pd.DataFrame:
    path = _find(RAW_DIR, "Monthly-AE")
    print(f"[A&E] Reading {path.name}...")

    df = pd.read_excel(path, sheet_name="Activity", header=13, skiprows=0)
    df = _normalise_cols(df)
    print(f"[A&E] Columns: {list(df.columns[:8])}")

    # The date column is 'period' or the first non‑unnamed column containing dates
    date_col = _pick_col(df, ["period", "unnamed_1"])
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Drop leading blank column (index 0) if present
    if df.columns[0] != "date":
        df = df.drop(columns=[df.columns[0]], errors="ignore")

    # Coerce remaining columns to numeric
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    tidy = (
        df[["date"] + num_cols]
        .melt(id_vars="date", var_name="metric", value_name="value")
        .dropna(subset=["value"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    tidy["source"] = "NHS England A&E"

    _save(tidy, "ae_clean.csv")
    print(f"[A&E] Done — {len(tidy)} rows, {tidy['metric'].nunique()} metrics")
    return tidy


# ---------------------------------------------------------------------------
# 3. Bed Occupancy
#    Combines the old KH03 CSV (2010‑2024) with seven quarterly XLSX files
#    (Q2 2024/25 → Q4 2025/26) to extend the series to 2026‑01‑01.
# ---------------------------------------------------------------------------
def clean_beds() -> pd.DataFrame:
    # --- part 1: old KH03 CSV ---
    path = _find(RAW_DIR, "KH03")
    print(f"[Beds] Reading {path.name}...")

    df = pd.read_csv(path, encoding="utf-8", low_memory=False, thousands=",")
    df = _normalise_cols(df)

    df["date"] = pd.to_datetime(df["effective_snapshot_date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["number_of_beds"] = pd.to_numeric(df["number_of_beds"], errors="coerce")

    national = (
        df.groupby("date", as_index=False)["number_of_beds"]
        .sum()
        .rename(columns={"number_of_beds": "value"})
    )
    national["metric"] = "total_occupied_beds_overnight"
    national["source"] = "NHS England KH03"

    # --- part 2: new quarterly XLSX files (2024‑2026) ---
    quarterly_files = [
        ("2024-07-01", "Beds-Open-Overnight-Web_File-Q2-2024-25.xlsx"),
        ("2024-10-01", "Beds-Open-Overnight-Web_File-Q3-2024-25-revised.xlsx"),
        ("2025-01-01", "Beds-Open-Overnight-Web_File-Q4-2024-25.xlsx"),
        ("2025-04-01", "Beds-Open-Overnight-Web_File-Q1-2025-26-revised.xlsx"),
        ("2025-07-01", "Beds-Open-Overnight-Web_File-Q2-2025-26.xlsx"),
        ("2025-10-01", "Beds-Open-Overnight-Web_File-Q3-2025-26.xlsx"),
        ("2026-01-01", "Beds-Open-Overnight-Web_File-Q4-2025-26.xlsx"),
    ]

    new_rows = []
    for quarter_date, fname in quarterly_files:
        file_path = RAW_DIR / fname
        if not file_path.exists():
            print(f"   [WARNING] {fname} not found – skipping.")
            continue

        # Try the most likely sheet names in order
        xl = pd.ExcelFile(file_path)
        sheet = None
        for candidate in ["NHS Trust by Sector", "Region by Sector"]:
            if candidate in xl.sheet_names:
                sheet = candidate
                break
        if sheet is None:
            # fallback: use the second sheet (often the data sheet after a 'Data Quality' tab)
            sheet = xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0]

        # Read without assuming a header – find the real header row that contains "Org Name"
        raw_df = pd.read_excel(file_path, sheet_name=sheet, header=None)
        header_row = None
        for i, row in raw_df.iterrows():
            if row.astype(str).str.contains("Org Name", na=False).any():
                header_row = i
                break
        if header_row is None:
            print(f"   [WARNING] Could not find 'Org Name' header row in {fname} – skipping.")
            continue

        # Re‑read with the correct header
        df_q = pd.read_excel(file_path, sheet_name=sheet, header=header_row)
        cols = [str(c).strip() for c in df_q.columns]
        df_q.columns = cols

        # Find the England row
        england_rows = df_q[df_q["Org Name"].astype(str).str.strip().str.lower() == "england"]
        if england_rows.empty:
            print(f"   [WARNING] England row not found in {fname} – skipping.")
            continue
        eng_row = england_rows.iloc[0]

        # Find the second 'Total' column (occupied beds)
        total_cols = [i for i, c in enumerate(cols) if c.lower().startswith("total")]
        if len(total_cols) < 2:
            print(f"   [WARNING] Could not identify the two 'Total' columns in {fname} – skipping.")
            continue
        occ_col = cols[total_cols[1]]
        total_val = pd.to_numeric(eng_row[occ_col], errors="coerce")
        if pd.isna(total_val):
            print(f"   [WARNING] Occupied beds value is NaN in {fname} – skipping.")
            continue

        new_rows.append({
            "date": pd.to_datetime(quarter_date),
            "value": float(total_val),
            "metric": "total_occupied_beds_overnight",
            "source": "NHS England KH03",
        })

    new_df = pd.DataFrame(new_rows)
    if not new_df.empty:
        # Avoid duplicating quarters that already exist in the old data
        national = national[~national["date"].isin(new_df["date"])]
        combined = pd.concat([national, new_df], ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)
    else:
        combined = national

    _save(combined, "beds_clean.csv")
    print(f"[Beds] Done — {len(combined)} time points (now ends {combined['date'].max().strftime('%Y-%m-%d')})")
    return combined


# ---------------------------------------------------------------------------
# 4. ONS Population
#    File: ons_uk_population.csv – rows 0‑7 are metadata, data starts where
#    column 0 is a four‑digit year.
# ---------------------------------------------------------------------------
def clean_population() -> pd.DataFrame:
    path = RAW_DIR / "ons_uk_population.csv"
    if not path.exists():
        raise FileNotFoundError(f"ONS population file not found at {path}")
    print(f"[ONS] Reading {path.name}...")

    raw = pd.read_csv(path, header=None, encoding="utf-8")

    # Locate the first row where the first cell is a four‑digit year
    data_start = 0
    for i, row in raw.iterrows():
        val = str(row.iloc[0]).strip()
        if val.isdigit() and len(val) == 4:
            data_start = i
            break

    df = raw.iloc[data_start:].copy()
    df.columns = ["year", "population"] + list(range(len(df.columns) - 2))
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["population"] = pd.to_numeric(
        df["population"].astype(str).str.replace(",", ""), errors="coerce"
    )
    df = df.dropna(subset=["year", "population"])
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01")

    out = df[["date", "population"]].sort_values("date").reset_index(drop=True)
    _save(out, "population_clean.csv")
    print(f"[ONS] Done — {len(out)} years of population data")
    return out


# ---------------------------------------------------------------------------
# 5. GP Appointments
#    File: National_Overview.csv – flat, clean CSV with a header on row 0.
# ---------------------------------------------------------------------------
def clean_gp() -> pd.DataFrame:
    path = RAW_DIR / "National_Overview.csv"
    if not path.exists():
        raise FileNotFoundError(f"GP appointments file not found at {path}")
    print(f"[GP] Reading {path.name}...")

    df = pd.read_csv(path)
    df = _normalise_cols(df)

    df["date"] = pd.to_datetime(
        df["appointment_month"].str.strip().str.capitalize(),
        format="%b%Y", errors="coerce",
    )
    df = df.dropna(subset=["date"])

    attended = df[df["appt_status"].str.lower() == "attended"].copy()
    attended["appointments"] = pd.to_numeric(attended["appointments"], errors="coerce")

    total = (
        attended.groupby("date", as_index=False)["appointments"]
        .sum()
        .rename(columns={"appointments": "value"})
    )
    total["metric"] = "total_attended_appointments"

    by_mode = (
        attended.groupby(["date", "appt_mode"], as_index=False)["appointments"]
        .sum()
        .rename(columns={"appointments": "value"})
    )
    by_mode["metric"] = "attended_" + (
        by_mode["appt_mode"].str.lower().str.replace(r"[\s/\-]+", "_", regex=True)
    )
    by_mode = by_mode.drop(columns=["appt_mode"])

    tidy = pd.concat([total, by_mode], ignore_index=True)
    tidy = tidy.sort_values(["metric", "date"]).reset_index(drop=True)
    tidy["source"] = "NHS Digital GP Appointments Data"

    _save(tidy, "gp_appointments_clean.csv")
    print(f"[GP] Done — {len(tidy)} rows, {tidy['metric'].nunique()} metrics")
    return tidy


# ---------------------------------------------------------------------------
# 6. NHS Workforce (HCHS)
#    File: *Workforce*.xlsx, sheet '1' (HCHS staff by staff group, monthly).
#    Header is row 4.  Columns from index 3 onward are month‑date headers.
# ---------------------------------------------------------------------------
def clean_workforce() -> pd.DataFrame:
    path = _find(RAW_DIR, "Workforce")
    print(f"[Workforce] Reading {path.name}, sheet '1'...")

    df = pd.read_excel(path, sheet_name="1", header=4)
    df = df.rename(columns={
        df.columns[0]: "data_type",
        df.columns[1]: "main_staff_group",
        df.columns[2]: "staff_group",
    })

    # Identify date columns (they parse as real dates)
    date_cols = []
    rename_map = {}
    for col in df.columns[3:]:
        dt = pd.to_datetime(col, errors="coerce")
        if pd.notna(dt):
            date_cols.append(col)
            rename_map[col] = dt

    df = df.rename(columns=rename_map)
    keep_cols = ["data_type", "main_staff_group", "staff_group"] + [
        rename_map.get(c, c) for c in date_cols
    ]
    df = df[keep_cols]

    tidy = df.melt(
        id_vars=["data_type", "main_staff_group", "staff_group"],
        var_name="date",
        value_name="value",
    )
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
    tidy = tidy.dropna(subset=["date"])
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])

    # Construct a descriptive metric name from the classification columns
    tidy["metric"] = (
        tidy["data_type"].astype(str).str.strip() + ": "
        + tidy["main_staff_group"].astype(str).str.strip()
        + " - " + tidy["staff_group"].astype(str).str.strip()
    )
    tidy = tidy[["date", "metric", "value"]].sort_values(["metric", "date"]).reset_index(drop=True)
    tidy["source"] = "NHS England HCHS Workforce Statistics"

    _save(tidy, "workforce_clean.csv")
    print(f"[Workforce] Done — {len(tidy)} rows, {tidy['metric'].nunique()} metrics")
    return tidy


# ---------------------------------------------------------------------------
# 7. HMT PESA — Chapter 4 (health & public expenditure)
# ---------------------------------------------------------------------------
def _clean_pesa_total_sheet(path: Path) -> pd.DataFrame:
    """Extract the 'Total public sector current expenditure' table from sheet 4_1."""
    raw = pd.read_excel(path, sheet_name="4_1", header=None)
    data = raw.iloc[5:].copy()
    data = data.rename(columns={
        1: "fiscal_year", 2: "nominal_gbp_bn", 3: "real_gbp_bn", 4: "pct_gdp",
    })
    data = data[["fiscal_year", "nominal_gbp_bn", "real_gbp_bn", "pct_gdp"]]
    data["fiscal_year"] = data["fiscal_year"].astype(str).str.strip()
    data = data[data["fiscal_year"].str.match(r"^\d{4}-\d{2}$", na=False)]

    for col in ["nominal_gbp_bn", "real_gbp_bn", "pct_gdp"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    tidy = data.melt(id_vars="fiscal_year", var_name="value_type", value_name="value")
    tidy = tidy.dropna(subset=["value"])
    tidy["category"] = "Total public sector current expenditure"
    return tidy


def _clean_pesa_function_sheet(path: Path, sheet_name: str, value_type: str) -> pd.DataFrame:
    """Extract function‑level spending from a PESA chapter 4 sheet."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    year_row = raw.iloc[4]
    year_cols = {
        i: str(v).strip() for i, v in year_row.items()
        if i > 0 and re.match(r"^\d{4}-\d{2}$", str(v).strip())
    }

    data = raw.iloc[5:].copy()
    data = data.rename(columns={0: "category"})
    data["category"] = data["category"].astype(str).str.strip()
    data = data[data["category"].notna() & (data["category"] != "nan") & (data["category"] != "")]

    keep_cols = ["category"] + list(year_cols.keys())
    data = data[keep_cols]
    data = data.rename(columns=year_cols)

    tidy = data.melt(id_vars="category", var_name="fiscal_year", value_name="value")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])
    tidy["value_type"] = value_type
    return tidy


def clean_pesa() -> pd.DataFrame:
    path = _find(RAW_DIR, "PESA")
    print(f"[PESA] Reading {path.name}...")

    total = _clean_pesa_total_sheet(path)

    functions = pd.concat([
        _clean_pesa_function_sheet(path, "4_2", "nominal_gbp_bn"),
        _clean_pesa_function_sheet(path, "4_3", "real_gbp_bn"),
        _clean_pesa_function_sheet(path, "4_4", "pct_gdp"),
    ], ignore_index=True)

    total["fiscal_year_clean"] = total["fiscal_year"]
    functions["fiscal_year_clean"] = functions["fiscal_year"]

    combined = pd.concat([
        total[["category", "fiscal_year_clean", "value_type", "value"]],
        functions[["category", "fiscal_year_clean", "value_type", "value"]],
    ], ignore_index=True)
    combined = combined.rename(columns={"fiscal_year_clean": "fiscal_year"})

    combined["date"] = pd.to_datetime(
        combined["fiscal_year"].str.slice(0, 4) + "-04-01", errors="coerce"
    )
    combined = combined.dropna(subset=["date"])
    combined["metric"] = combined["category"] + " (" + combined["value_type"] + ")"

    tidy = (
        combined[["date", "metric", "value"]]
        .sort_values(["metric", "date"])
        .reset_index(drop=True)
    )
    tidy["source"] = "HM Treasury PESA 2025, Chapter 4"

    _save(tidy, "pesa_clean.csv")
    print(f"[PESA] Done — {len(tidy)} rows, {tidy['metric'].nunique()} metrics")
    print("[PESA] Tip: filter metric.str.contains('Health') for the health-specific series")
    return tidy


# ---------------------------------------------------------------------------
# 8. NICE QALY cost‑effectiveness threshold history
# ---------------------------------------------------------------------------
def clean_nice() -> pd.DataFrame:
    path = RAW_DIR / "nice_qaly_threshold_history.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"NICE reference table not found at {path}. Run nice_reference.py first."
        )
    print(f"[NICE] Reading {path.name}...")

    df = pd.read_csv(path)
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df = df.rename(columns={"effective_date": "date"})

    out = df.sort_values("date").reset_index(drop=True)
    _save(out, "nice_clean.csv")
    print(f"[NICE] Done — {len(out)} policy events")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find(directory: Path, keyword: str) -> Path:
    """
    Return the most recently modified file in `directory` whose name
    contains `keyword`.  Used to pick up the latest version of raw data
    without hard‑coding filenames.
    """
    matches = list(directory.glob(f"*{keyword}*"))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '*{keyword}*' in {directory}. Run scraper first."
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names: lower‑case, replace spaces and punctuation with underscores."""
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def _pick_col(df: pd.DataFrame, candidates: list) -> str:
    """Find the first column whose name matches one of the candidate strings."""
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        matches = [col for col in df.columns if c in col]
        if matches:
            return matches[0]
    raise ValueError(
        f"Could not find column. Looked for: {candidates}. "
        f"Columns present: {list(df.columns)}"
    )


def _save(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame as a CSV in the processed directory."""
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    print(f"  → Saved {out}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all() -> dict:
    """Execute every cleaning function and return their outputs."""
    results = {}
    cleaners = {
        "rtt": clean_rtt,
        "ae": clean_ae,
        "beds": clean_beds,
        "population": clean_population,
        "gp": clean_gp,
        "workforce": clean_workforce,
        "pesa": clean_pesa,
        "nice": clean_nice,
    }
    for name, fn in cleaners.items():
        try:
            results[name] = fn()
            print(f"✓ {name}\n")
        except FileNotFoundError as e:
            print(f"⚠ {name}: {e}\n")
        except Exception as e:
            print(f"✗ {name}: {e}\n")
            import traceback; traceback.print_exc()
    return results


if __name__ == "__main__":
    run_all()