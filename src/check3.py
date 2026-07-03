import pandas as pd


df2 = pd.read_csv(r"C:\Users\44782\Desktop\empirical project\data\processed\dashboard_forecasts.csv")

ae_breach2 = df2[
    (df2["metric"] == "A&E 12-hour decisions to admit (breach flow)")
    & (df2["type"] == "history")
    & (df2["quarter"].between("2020-10-01", "2021-04-01"))
]
print(ae_breach2[["quarter", "value"]])