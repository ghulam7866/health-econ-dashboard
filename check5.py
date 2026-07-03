fc = res.get_forecast(steps=horizons, exog=exog_future)

if display_name == "A&E 12-hour decisions to admit (breach flow)":
    print(f"\n   [DEBUG] Raw forecast (unscaled) for {display_name}:")
    print(fc.predicted_mean)
    print(f"   [DEBUG] Scale factor: {scale}")
    print(f"   [DEBUG] exog_future values:\n{exog_future}")

mean_fc = fc.predicted_mean * scale
ci = fc.conf_int(alpha=0.05) * scale