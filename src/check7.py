import pandas as pd

INPUT_FILE = r"C:\Users\44782\Desktop\empirical project\data\processed\combined_quarterly.csv"

df = pd.read_csv(INPUT_FILE)
df["quarter"] = pd.to_datetime(df["quarter"])

# covid_pulse should exist as a column already aligned to the quarterly index
# adjust this if it's stored differently (e.g. per-metric long format)
cols_to_check = ["quarter", "covid_pulse"]
if "post_covid_regime" in df.columns:
    cols_to_check.append("post_covid_regime")
if "post_covid_trend_break" in df.columns:
    cols_to_check.append("post_covid_trend_break")

sub = df[cols_to_check].drop_duplicates().sort_values("quarter")
print(sub.to_string(index=False))