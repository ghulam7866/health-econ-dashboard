"""
validate_forecasts.py
---------------------
Validates forecast outputs for:
1. Spikes and drops (sudden large changes)
2. Uniform/linear patterns (too smooth)
3. Extreme values (out of reasonable bounds)
4. Trend reversals (abrupt direction changes)
5. Plausibility checks against historical data

Usage:
    python validate_forecasts.py

Output:
    - Validation report -> <REPORT_DIR>/forecast_validation_report.txt
    - Summary CSV -> <REPORT_DIR>/forecast_validation_summary.csv
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_DIR = r"C:\Users\44782\Desktop\empirical project"
FORECAST_FILE = os.path.join(PROJECT_DIR, "data", "processed", "dashboard_forecasts.csv")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
PLOTS_DIR = os.path.join(REPORT_DIR, "validation_plots")

# Create plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)

# Metrics to validate
METRICS_TO_VALIDATE = [
    "RTT waiting list (level)",
    "A&E attendances (flow)",
    "Workforce FTE (level)",
    "Bed occupancy (level)",
    "RTT % within 18 weeks (performance)",
    "A&E 12-hour decisions to admit (breach flow)",
]

# Skip GP series (no forecasts)
SKIP_METRICS = [
    "GP total appointments (flow)",
    "GP face-to-face appointments (flow)", 
    "GP telephone appointments (flow)",
]

# ---------------------------------------------------------------------------
# Tee logger
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def detect_spikes(series, threshold=3.0):
    """
    Detect spikes and drops using z-score method.
    threshold: number of standard deviations to flag
    """
    if len(series) < 3:
        return []
    
    # Calculate differences
    diffs = np.diff(series)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    if std_diff == 0:
        return []
    
    z_scores = np.abs((diffs - mean_diff) / std_diff)
    spike_indices = np.where(z_scores > threshold)[0]
    
    return [
        {
            "position": i,
            "value": series[i],
            "prev_value": series[i-1] if i > 0 else None,
            "change": diffs[i],
            "z_score": z_scores[i]
        }
        for i in spike_indices
    ]


def detect_uniform_pattern(series, tolerance=0.02):
    """
    Detect if a series is too uniform (linear/mechanical).
    tolerance: how much variation is allowed to be considered uniform
    """
    if len(series) < 4:
        return False
    
    # Check if series is almost linear (correlation with time > 0.99)
    t = np.arange(len(series))
    corr = np.corrcoef(series, t)[0, 1]
    
    if abs(corr) > 0.99:
        return {
            "is_uniform": True,
            "correlation": corr,
            "reason": "Almost perfectly linear (correlation > 0.99)"
        }
    
    # Check if changes are constant (low variance in differences)
    diffs = np.diff(series)
    if len(diffs) > 2:
        diff_cv = np.std(diffs) / (np.abs(np.mean(diffs)) + 1e-6)
        if diff_cv < tolerance:
            return {
                "is_uniform": True,
                "correlation": corr,
                "reason": f"Constant rate of change (CV={diff_cv:.3f})"
            }
    
    return {"is_uniform": False}


def detect_extreme_values(series, historical_stats, metric_name):
    """
    Detect if forecast values are extreme compared to historical data.
    """
    if historical_stats is None:
        return []
    
    extremes = []
    hist_min, hist_max, hist_mean, hist_std = historical_stats
    
    for i, val in enumerate(series):
        # Check if value is outside [min, max]
        if val < hist_min * 0.5:  # Less than 50% of historical min
            extremes.append({
                "position": i,
                "value": val,
                "issue": f"Below historical min ({hist_min:.2f}) by {abs(val - hist_min):.2f}",
                "severity": "high" if val < hist_min * 0.3 else "medium"
            })
        elif val > hist_max * 1.5:  # More than 150% of historical max
            extremes.append({
                "position": i,
                "value": val,
                "issue": f"Above historical max ({hist_max:.2f}) by {abs(val - hist_max):.2f}",
                "severity": "high" if val > hist_max * 2 else "medium"
            })
        # Check if value is more than 3 std from mean
        elif abs(val - hist_mean) > 3 * hist_std:
            extremes.append({
                "position": i,
                "value": val,
                "issue": f"More than 3 std from mean ({hist_mean:.2f})",
                "severity": "medium"
            })
    
    return extremes


def detect_trend_reversals(series, window=3):
    """
    Detect abrupt trend reversals (direction changes).
    window: number of points to check for consistent direction
    """
    if len(series) < window * 2:
        return []
    
    reversals = []
    diffs = np.diff(series)
    directions = np.sign(diffs)
    
    for i in range(len(directions) - window):
        # Check if current window has one direction, next window has opposite
        if all(d == directions[i] for d in directions[i:i+window]) and \
           all(d == -directions[i] for d in directions[i+window:i+window*2]):
            reversals.append({
                "position": i + window,
                "value": series[i + window],
                "direction_change": f"{'up' if directions[i] > 0 else 'down'} to {'up' if directions[i] < 0 else 'down'}"
            })
    
    return reversals


def detect_smoothness(series, max_linear_segments=3):
    """
    Detect if a forecast is too smooth (like a polynomial).
    """
    if len(series) < 6:
        return None
    
    # Fit polynomial of degree 2
    t = np.arange(len(series))
    coeffs = np.polyfit(t, series, 2)
    fitted = np.polyval(coeffs, t)
    
    # Calculate R-squared
    ss_res = np.sum((series - fitted) ** 2)
    ss_tot = np.sum((series - np.mean(series)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    # If R-squared > 0.95, it's too smooth (polynomial-like)
    if r2 > 0.95:
        return {
            "is_too_smooth": True,
            "r_squared": r2,
            "reason": f"Almost perfectly fits quadratic (R²={r2:.3f})"
        }
    
    return {"is_too_smooth": False}


def validate_metric(df, metric_name):
    """
    Run all validation checks on a single metric.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING: {metric_name}")
    print(f"{'='*60}")
    
    # Get metric data
    metric_df = df[df["metric"] == metric_name].sort_values("quarter")
    hist_df = metric_df[metric_df["type"] == "history"]
    fore_df = metric_df[metric_df["type"] == "forecast"]
    
    if hist_df.empty:
        print(f"  WARNING: No historical data found for {metric_name}")
        return None
    
    if fore_df.empty:
        print(f"  SKIP: No forecasts found for {metric_name}")
        return None
    
    # Extract values
    hist_values = hist_df["value"].values
    fore_values = fore_df["value"].values
    
    print(f"  Historical: {len(hist_values)} points")
    print(f"  Forecast: {len(fore_values)} points")
    print(f"  Last historical: {hist_values[-1]:.2f}")
    print(f"  First forecast: {fore_values[0]:.2f}")
    print(f"  Last forecast: {fore_values[-1]:.2f}")
    print(f"  Max forecast: {fore_values.max():.2f}")
    print(f"  Min forecast: {fore_values.min():.2f}")
    
    # Calculate historical stats
    hist_stats = (
        hist_values.min(),
        hist_values.max(),
        hist_values.mean(),
        hist_values.std(),
    )
    
    # 1. Detect spikes
    spikes = detect_spikes(fore_values)
    if spikes:
        print(f"\n  ⚠️ SPIKES/DROPS detected ({len(spikes)}):")
        for spike in spikes:
            print(f"    - Position {spike['position']}: {spike['value']:.2f} "
                  f"(change: {spike['change']:+.2f}, z-score: {spike['z_score']:.2f})")
    else:
        print(f"\n  ✅ No spikes/drops detected")
    
    # 2. Detect uniform patterns
    uniform = detect_uniform_pattern(fore_values)
    if uniform["is_uniform"]:
        print(f"\n  ⚠️ UNIFORM PATTERN detected:")
        print(f"    - {uniform['reason']}")
        print(f"    - Correlation with time: {uniform.get('correlation', 0):.3f}")
    else:
        print(f"\n  ✅ No uniform pattern detected")
    
    # 3. Detect extreme values
    extremes = detect_extreme_values(fore_values, hist_stats, metric_name)
    if extremes:
        print(f"\n  ⚠️ EXTREME VALUES detected ({len(extremes)}):")
        for extreme in extremes:
            print(f"    - Position {extreme['position']}: {extreme['value']:.2f} "
                  f"({extreme['issue']}) [Severity: {extreme['severity']}]")
    else:
        print(f"\n  ✅ No extreme values detected")
    
    # 4. Detect trend reversals
    reversals = detect_trend_reversals(fore_values)
    if reversals:
        print(f"\n  ⚠️ TREND REVERSALS detected ({len(reversals)}):")
        for rev in reversals[:5]:  # Show first 5
            print(f"    - Position {rev['position']}: {rev['value']:.2f} "
                  f"({rev['direction_change']})")
        if len(reversals) > 5:
            print(f"    - ... and {len(reversals) - 5} more")
    else:
        print(f"\n  ✅ No abrupt trend reversals detected")
    
    # 5. Detect smoothness
    smoothness = detect_smoothness(fore_values)
    if smoothness and smoothness.get("is_too_smooth", False):
        print(f"\n  ⚠️ TOO SMOOTH detected:")
        print(f"    - {smoothness['reason']}")
    else:
        print(f"\n  ✅ Forecast has natural variation")
    
    # 6. Plausibility checks
    print(f"\n  📊 PLAUSIBILITY CHECKS:")
    
    # Check if forecast follows recent trend
    last_hist = hist_values[-1]
    first_fore = fore_values[0]
    last_fore = fore_values[-1]
    
    # Percent change from last historical to first forecast
    change_initial = ((first_fore - last_hist) / last_hist) * 100
    print(f"    - Change from last historical to first forecast: {change_initial:+.1f}%")
    
    # Percent change from first to last forecast
    change_total = ((last_fore - first_fore) / first_fore) * 100
    print(f"    - Change from first to last forecast: {change_total:+.1f}%")
    
    # Check if forecast stays within reasonable range
    if abs(change_initial) > 30:
        print(f"    ⚠️ LARGE INITIAL CHANGE: {change_initial:+.1f}% (may indicate issue)")
    elif abs(change_initial) > 15:
        print(f"    ⚠️ MODERATE INITIAL CHANGE: {change_initial:+.1f}% (review recommended)")
    else:
        print(f"    ✅ Initial change is reasonable")
    
    # Check for unrealistic oscillation
    oscillation = np.std(np.diff(fore_values)) / (np.abs(np.mean(fore_values)) + 1e-6)
    if oscillation > 0.15:
        print(f"    ⚠️ HIGH OSCILLATION: coefficient of variation in changes = {oscillation:.3f}")
    else:
        print(f"    ✅ Forecast changes are stable")
    
    # Compile results
    return {
        "metric": metric_name,
        "n_hist": len(hist_values),
        "n_fore": len(fore_values),
        "last_hist": last_hist,
        "first_fore": first_fore,
        "last_fore": last_fore,
        "change_initial_pct": change_initial,
        "change_total_pct": change_total,
        "spike_count": len(spikes),
        "uniform_detected": uniform["is_uniform"],
        "extreme_count": len(extremes),
        "reversal_count": len(reversals),
        "too_smooth": smoothness.get("is_too_smooth", False) if smoothness else False,
        "oscillation": oscillation,
        "max_fore": fore_values.max(),
        "min_fore": fore_values.min(),
        "overall_score": "PASS" if (len(spikes) == 0 and 
                                     not uniform["is_uniform"] and 
                                     len(extremes) == 0 and 
                                     abs(change_initial) < 30) else "REVIEW"
    }


def plot_forecast_validation(metric_name, hist_values, fore_values, issues, plots_dir):
    """
    Create a validation plot for a metric.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top plot: Time series with forecast
    ax1 = axes[0]
    ax1.plot(range(len(hist_values)), hist_values, 'b-', label='Historical', linewidth=2)
    ax1.plot(range(len(hist_values), len(hist_values) + len(fore_values)), 
             fore_values, 'r-', label='Forecast', linewidth=2, linestyle='--')
    
    # Highlight issues
    for issue in issues.get('spikes', []):
        pos = issue['position'] + len(hist_values)
        ax1.axvline(x=pos, color='orange', alpha=0.3, linestyle=':')
    
    for issue in issues.get('extremes', []):
        pos = issue['position'] + len(hist_values)
        ax1.axvline(x=pos, color='red', alpha=0.3, linestyle='--')
    
    ax1.axvline(x=len(hist_values) - 0.5, color='gray', linestyle='--', alpha=0.5, label='Forecast start')
    ax1.set_title(f'{metric_name} - Validation')
    ax1.set_xlabel('Time (quarters)')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Change analysis
    ax2 = axes[1]
    full_series = np.concatenate([hist_values, fore_values])
    changes = np.diff(full_series)
    
    ax2.bar(range(len(changes)), changes, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=len(hist_values) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Change Analysis (quarter-over-quarter)')
    ax2.set_xlabel('Time (quarters)')
    ax2.set_ylabel('Change')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    safe_name = metric_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    plot_path = os.path.join(plots_dir, f"validation_{safe_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    print("=" * 70)
    print("FORECAST VALIDATION SCRIPT")
    print("=" * 70)
    print(f"Run started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(REPORT_DIR, f"forecast_validation_log_{timestamp}.txt")
    summary_path = os.path.join(REPORT_DIR, f"forecast_validation_summary_{timestamp}.csv")
    sys.stdout = Tee(log_path)
    
    # Load forecast data
    df = pd.read_csv(FORECAST_FILE)
    df["quarter"] = pd.to_datetime(df["quarter"])
    
    print(f"\nLoaded {len(df)} forecast records")
    print(f"Metrics: {df['metric'].unique().tolist()}")
    
    # Validate each metric
    results = []
    for metric_name in METRICS_TO_VALIDATE:
        result = validate_metric(df, metric_name)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    summary_df = pd.DataFrame(results)
    
    # Format for display
    display_cols = ['metric', 'n_hist', 'n_fore', 'change_initial_pct', 
                    'change_total_pct', 'spike_count', 'uniform_detected', 
                    'extreme_count', 'reversal_count', 'overall_score']
    
    print("\n" + summary_df[display_cols].to_string(index=False))
    
    # Save summary
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Full log saved to: {log_path}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    metrics_to_review = summary_df[summary_df["overall_score"] == "REVIEW"]
    if not metrics_to_review.empty:
        print(f"\n⚠️ {len(metrics_to_review)} metrics need review:")
        for _, row in metrics_to_review.iterrows():
            issues = []
            if row['spike_count'] > 0:
                issues.append(f"{row['spike_count']} spikes")
            if row['uniform_detected']:
                issues.append("uniform pattern")
            if row['extreme_count'] > 0:
                issues.append(f"{row['extreme_count']} extremes")
            if row['reversal_count'] > 0:
                issues.append(f"{row['reversal_count']} reversals")
            if row['too_smooth']:
                issues.append("too smooth")
            print(f"  ❌ {row['metric']}: {', '.join(issues)}")
    else:
        print("\n✅ All metrics passed validation!")
    
    print("\n" + "=" * 70)
    print("Run completed.")


if __name__ == "__main__":
    main()