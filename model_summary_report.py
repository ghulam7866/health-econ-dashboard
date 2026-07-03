# summary_report.py
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
df = pd.read_csv(PROCESSED_DIR / "dashboard_forecasts.csv")

summary = df.groupby("metric").agg({
    "type": lambda x: (x == "forecast").sum(),
    "value": ["min", "max", "mean"]
})
print(summary)