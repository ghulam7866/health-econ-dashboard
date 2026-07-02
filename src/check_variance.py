import pandas as pd

# Load the newly generated asset
forecast_file = r"C:\Users\44782\Desktop\empirical project\data\processed\forecast_6yr.csv"
df = pd.read_csv(forecast_file)

print("=" * 60)
print("             FORECAST SLOPE VARIANCE CHECK")
print("=" * 60)

# Filter out only future projections
forecast_only = df[df["type"] == "forecast"]

for metric in forecast_only["metric"].unique():
    metric_df = forecast_only[forecast_only["metric"] == metric].sort_values("quarter")
    
    # Calculate variance / standard deviation across the 24 quarters
    f_std = metric_df["value"].std()
    start_val = metric_df["value"].iloc[0]
    end_val = metric_df["value"].iloc[-1]
    net_change = end_val - start_val
    
    if f_std == 0:
        print(f"❌ {metric:<40} -> STALLED (Perfect Flatline, Std=0.0)")
    else:
        print(f"✓ {metric:<40} -> DYNAMIC (Std={f_std:,.2f} | Shift: {start_val:,.1f} ➔ {end_val:,.1f})")

print("=" * 60)