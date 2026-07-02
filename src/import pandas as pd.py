import pandas as pd

df = pd.read_csv(r"C:\Users\44782\Desktop\empirical project\data\processed\forecast_6yr.csv")

gp_series = [
    "GP total appointments (flow)",
    "GP face-to-face appointments (flow)",
    "GP telephone appointments (flow)",
]

hits = df[df["metric"].isin(gp_series)]
print(len(hits), "rows found")
print(hits["metric"].unique())
print(hits.head(10))