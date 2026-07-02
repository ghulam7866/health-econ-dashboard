"""
validate_forecasts.py
---------------------
Validates forecast outputs for quality and plausibility.

This script checks forecasts for:
1. Spikes and drops (sudden large changes)
2. Uniform/linear patterns (too smooth/mechanical)
3. Extreme values (outside reasonable bounds)
"""