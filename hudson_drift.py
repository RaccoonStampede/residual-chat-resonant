#!/usr/bin/env python3
"""
Hudson's Drift Law — Macro Inefficiency Auditor + Harness Edition
Deterministic, residual-compatible implementation.
Standalone analytic reference (LanguageCore has its own native controller).
"""

import numpy as np
from typing import Dict, Optional

def hudsons_drift_law(
    n: float,
    epsilon: float = 0.05,
    a: float = 0.001,
    sigma: float = 0.15,
    alpha: float = 1.8,
    beta: float = 0.2,
    gamma: float = 0.05,
) -> float:
    if abs(epsilon) < 1e-12:
        return 0.0
    cap = np.sign(epsilon) * np.sqrt(np.abs(epsilon) / (2.0 * a))
    cap_eff = cap / (1.0 + max(0.0, gamma))
    tau = max(sigma ** 2 / abs(epsilon), 1.0)
    fractal_term = 1.0 - np.exp(-(n ** alpha) / tau)
    chaotic_term = beta * np.sin(gamma * n)
    return float(cap_eff * fractal_term + chaotic_term)

def residual_scale_from_gamma(gamma: float, base_lock: float = 0.888) -> float:
    return float(base_lock + (1.0 - base_lock) / (1.0 + gamma / 50.0))

def recommended_gamma_for_lock(current_drift: float, target_drift: float = 0.05) -> float:
    if current_drift <= target_drift:
        return 50.0
    ratio = min(20.0, current_drift / max(target_drift, 1e-6))
    return float(50.0 * ratio)