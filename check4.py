import pandas as pd


df = pd.read_csv(r"C:\Users\44782\Desktop\empirical project\data\processed\ae_clean.csv")
df["date"] = pd.to_datetime(df["date"])

breach = df[
    (df["metric"] == "number_of_patients_spending_12_hours_from_decision_to_admit_to_admission")
].sort_values("date")

print(breach[["date", "value"]].to_string())