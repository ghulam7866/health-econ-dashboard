import pandas as pd

df = pd.read_csv(r"C:\Users\44782\Desktop\empirical project\data\processed\dashboard_forecasts.csv")

breach = df[
    (df["metric"] == "A&E 12-hour decisions to admit (breach flow)")
    & (df["type"] == "forecast")
].sort_values("quarter")

print(breach[["quarter", "value"]].to_string())