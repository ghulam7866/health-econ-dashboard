"""
select_arima_orders.py
------------------------
Uses pmdarima's auto_arima to search (p,d,q)(P,D,Q,4) orders per series,
with the exog dummies from exog_config.py included so the search reflects
the model we'll actually fit - not a no-exog approximation.

d and D are FIXED based on our own diagnostics rather than left to
auto_arima's internal tests:
    - d=1 for all three series (confirmed via ADF/KPSS + overdifferencing
      check for workforce)
    - D=1 (seasonal difference) is offered as a candidate but auto_arima's
      seasonal test (OCSB) decides whether it's actually used - quarterly
      NHS data usually has real seasonality (winter A&E pressure, etc.)
      so this is worth letting the test decide rather than forcing it.

auto_arima minimises AICc by default - a reasonable criterion for our
sample sizes (60-76 obs), where AIC alone can overfit slightly more.

Run:
    pip install pmdarima
    python src/select_arima_orders.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from pmdarima import auto_arima

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exog_config import EXOG_CONFIG, METRIC_NAMES

PROCESSED_DIR = Path(r"C:\Users\44782\Desktop\empirical project\data\processed")
COMBINED_PATH = PROCESSED_DIR / "combined_quarterly.csv"

# Fixed d per series (overrides auto_arima's own unit-root test), based on
# our own ADF/KPSS + overdifferencing diagnostics
FIXED_D = {
    "RTT waiting list (level)": 1,
    "A&E attendances (flow)": 1,
    "Workforce FTE (level)": 1,
}

SEASONAL_PERIOD = 4  # quarterly data -> annual seasonality


def select_for_series(df: pd.DataFrame, label: str):
    metric = METRIC_NAMES[label]
    exog_cols = EXOG_CONFIG[label]
    d = FIXED_D[label]

    sub = df[df["metric"] == metric].sort_values("quarter").reset_index(drop=True)
    y = sub["value"]
    X = sub[exog_cols] if exog_cols else None

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  n obs: {len(y)}  |  fixed d={d}  |  exog: {exog_cols or 'none'}")
    print(f"{'='*70}")

    model = auto_arima(
        y,
        X=X,
        d=d,
        seasonal=True,
        m=SEASONAL_PERIOD,
        stepwise=False,        # full grid search — stepwise can miss seasonal D
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aicc",
        trace=True,
        max_p=3, max_q=3,
        max_P=1, max_Q=1,
        max_D=1,               # explicitly allow seasonal differencing
        n_jobs=-1,              # parallelise the grid search
    )

    print(f"\n  SELECTED ORDER: {model.order}  seasonal_order: {model.seasonal_order}")
    print(f"  AICc: {model.aicc():.2f}")

    return {
        "metric": label,
        "order": model.order,
        "seasonal_order": model.seasonal_order,
        "aicc": model.aicc(),
        "exog_cols": exog_cols,
    }


def main():
    df = pd.read_csv(COMBINED_PATH)
    df["quarter"] = pd.to_datetime(df["quarter"])

    results = []
    for label in EXOG_CONFIG.keys():
        r = select_for_series(df, label)
        results.append(r)

    print(f"\n{'='*70}")
    print("SUMMARY — orders to use for final SARIMAX fit")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['metric']}")
        print(f"    order={r['order']}  seasonal_order={r['seasonal_order']}  "
              f"exog={r['exog_cols']}  AICc={r['aicc']:.1f}")

    out = pd.DataFrame(results)
    out_path = PROCESSED_DIR / "arima_order_selection.csv"
    out.to_csv(out_path, index=False)
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()