import pandas as pd

df = pd.read_csv(r"C:\Users\44782\Desktop\empirical project\data\processed\combined_quarterly.csv")

ae_breach = df[
    (df["metric"] == "number_of_patients_spending_12_hours_from_decision_to_admit_to_admission")
    & (df["quarter"].between("2020-10-01", "2021-04-01"))
]
pop = df[
    (df["metric"] == "uk_population")
    & (df["quarter"].between("2020-10-01", "2021-04-01"))
]

print(ae_breach[["quarter", "value"]])
print(pop[["quarter", "value"]])