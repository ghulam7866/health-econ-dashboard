import pandas as pd

df = pd.read_csv(r"C:\Users\44782\Desktop\empirical project\data\processed\dashboard_forecasts.csv")

gp_series = [
    "GP total appointments (flow)",
    "GP face-to-face appointments (flow)",
    "GP telephone appointments (flow)",
]

hits = df[df["metric"].isin(gp_series)]
print(len(hits), "rows found")