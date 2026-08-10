#!/usr/bin/env python3
"""
Residual-EGGROLL — Clean deterministic low-rank Evolution Strategies
on top of LanguageCore.

Fixes applied (2026-08-09):
- Rank-weight polarity corrected (best fitness now receives positive weight)
- Candidate isolation improved: coherent phase, void state/phase, and
  sequence buffer are also snapshotted and restored between population members.
"""

import numpy as np
from typing import Dict, List, Optional
import math

# Minimal stubs so this module can be imported and tested standalone.
# In the real stack these come from LanguageCore.
def det_vector(n: int, base_id: float, t: float, scale: float = 1.0) -> np.ndarray:
    return np.array([
        math.sin(base_id + i * 0.173 + t * 0.72) * 0.1 * scale
        for i in range(n)
    ], dtype=np.float64)

class LanguageCore:
    """Minimal stub for testing ResidualEGGROLL isolation + polarity."""
    def __init__(self, coherent_dim: int = 32, seed: int = 42):
        self.coherent_dim = coherent_dim
        self.base_id = float(seed) * 0.917
        self.coherent_state = det_vector(coherent_dim, self.base_id, 0.0, 0.42)
        self.coherent_phase = np.linspace(0.0, 2 * np.pi, coherent_dim, endpoint=False)
        self.void_dim = 40
        self.void_state = det_vector(self.void_dim, self.base_id + 30.0, 0.0, 0.11)
        self.void_phase = np.linspace(0.0, 2 * np.pi, self.void_dim, endpoint=False)
        self.seq_buffer = np.zeros(8)
        self.kernels = [np.eye(coherent_dim) * 0.12]
        self.steps = 0
        self.imprint_deep = self.coherent_state.copy() * 0.02
        self.coherent_frac_target = 0.08

    def phase_lock_metric(self, phases: np.ndarray) -> float:
        if len(phases) < 1:
            return 0.0
        return float(np.abs(np.mean(np.exp(1j * phases))))

    def hierarchical_locks(self):
        d = self.coherent_dim
        c = max(2, d // 3)
        b = max(2, d // 3)
        return (
            self.phase_lock_metric(self.coherent_phase[:c]),
            self.phase_lock_metric(self.coherent_phase[c:c+b]),
            self.phase_lock_metric(self.coherent_phase[c+b:]),
        )

    def coherent_fraction(self) -> float:
        c = np.linalg.norm(self.coherent_state) + 1e-12
        v = np.linalg.norm(self.void_state) + 1e-12
        return float(c / (c + v))

    def _enforce_fraction(self, hard: bool = False):
        pass

    def couple(self):
        # Intentionally mutates more than just coherent_state
        self.coherent_phase += 0.01 * np.sin(self.coherent_phase)
        self.void_state += 0.002 * np.mean(self.coherent_state)
        self.void_phase += 0.005
        self.seq_buffer = 0.95 * self.seq_buffer + 0.05 * np.mean(self.void_state[:4])
        self.steps += 1

    def step(self):
        self.couple()


class ResidualEGGROLL:
    """
    Low-rank residual population optimizer for LanguageCore coherent sector.
    """

    def __init__(
        self,
        core: Optional[LanguageCore] = None,
        rank: int = 4,
        pop_size: int = 16,
        sigma: float = 0.08,
        lr: float = 0.025,
        seed: int = 42,
    ):
        self.core = core if core is not None else LanguageCore(seed=seed)
        self.rank = rank
        self.pop_size = pop_size
        self.base_sigma = sigma
        self.sigma = sigma
        self.base_lr = lr
        self.base_id = float(seed) * 0.917 + 11.0
        self.generation = 0
        self.history: List[Dict] = []

        self.best_coherent = self.core.coherent_state.copy()
        self.best_fitness = -1e9
        self.best_lock = 0.0

        self.drift_history: List[float] = []
        self.smoothed_drift: Optional[float] = None
        self.calm_streak = 0
        self.lr_scale = 1.0
        self.min_lr_scale = 0.30
        self.max_lr_scale = 1.80
        self.lock_threshold = 8
        self.locked = False
        self.HUDSON_LOCK = 0.888

    def _low_rank_residual(self, member_id: int, t: float) -> np.ndarray:
        d = self.core.coherent_dim
        r = self.rank
        A = np.stack([
            det_vector(r, self.base_id + member_id * 3.7 + i * 0.19, t, self.sigma)
            for i in range(d)
        ])
        B = np.stack([
            det_vector(r, self.base_id + member_id * 3.7 + 517.0 + i * 0.23, t + 0.41, self.sigma)
            for i in range(d)
        ])
        E = (A @ B.T) / math.sqrt(max(1, r))
        return E

    def _current_drift(self) -> float:
        c = self.core.coherent_state
        return float(np.sum((c - np.mean(c)) ** 2))

    def update_drift_controller(self) -> float:
        drift = self._current_drift()
        self.drift_history.append(drift)
        if len(self.drift_history) > 20:
            self.drift_history.pop(0)

        avg_var = float(np.var(self.drift_history)) if len(self.drift_history) > 1 else drift
        if self.smoothed_drift is None:
            self.smoothed_drift = avg_var
        else:
            self.smoothed_drift = 0.88 * self.smoothed_drift + 0.12 * avg_var

        if self.smoothed_drift > 0.12:
            lr_mult = 0.88
            self.sigma = max(self.base_sigma * 0.45, self.sigma * 0.90)
            self.calm_streak = 0
        elif self.smoothed_drift > 0.06:
            lr_mult = 0.96
            self.sigma = max(self.base_sigma * 0.60, self.sigma * 0.97)
            self.calm_streak = max(0, self.calm_streak - 1)
        elif self.smoothed_drift < 0.035:
            self.calm_streak += 1
            lr_mult = 1.012
            self.sigma = min(self.base_sigma * 1.10, self.sigma * 1.008)
        else:
            self.calm_streak = max(0, self.calm_streak - 1)
            lr_mult = 1.0

        self.lr_scale = float(np.clip(self.lr_scale * lr_mult, self.min_lr_scale, self.max_lr_scale))

        lock = self.core.phase_lock_metric(self.core.coherent_phase)
        if (not self.locked) and self.calm_streak >= self.lock_threshold and lock > 0.97:
            self.locked = True
            self.lr_scale = min(self.lr_scale, self.HUDSON_LOCK)
            self.sigma = min(self.sigma, self.base_sigma * 0.70)

        if self.locked:
            self.lr_scale = min(self.lr_scale, self.HUDSON_LOCK)
            self.sigma = min(self.sigma, self.base_sigma * 0.75)

        return self.lr_scale

    def _fitness(self) -> float:
        core_l, bridge_l, edge_l = self.core.hierarchical_locks()
        hierarchical = 0.45 * core_l + 0.30 * bridge_l + 0.25 * edge_l
        frac = self.core.coherent_fraction()
        frac_err = abs(frac - self.core.coherent_frac_target)
        frac_score = max(0.0, 1.0 - frac_err * 8.0)
        deep_norm = float(np.linalg.norm(self.core.imprint_deep))
        deep_score = min(1.0, deep_norm * 4.0)
        drift = self._current_drift()
        drift_pen = max(0.0, 1.0 - drift * 3.5)
        fitness = 0.48 * hierarchical + 0.22 * frac_score + 0.15 * deep_score + 0.15 * drift_pen
        return float(fitness)

    def step(self, n_core_steps: int = 3) -> Dict:
        t = self.generation * 0.07
        d = self.core.coherent_dim

        residuals = []
        fitnesses = []

        # --- Full state snapshot for proper candidate isolation ---
        orig_coherent = self.core.coherent_state.copy()
        orig_phase = self.core.coherent_phase.copy()
        orig_void = self.core.void_state.copy()
        orig_void_phase = self.core.void_phase.copy()
        orig_seq = self.core.seq_buffer.copy()
        orig_kernels = [k.copy() for k in self.core.kernels]
        orig_steps = self.core.steps

        for i in range(self.pop_size):
            E = self._low_rank_residual(i, t)
            residuals.append(E)

            # Apply residual
            self.core.coherent_state = orig_coherent + 0.35 * np.mean(E, axis=1)
            self.core._enforce_fraction(hard=False)

            if len(self.core.kernels) > 0 and self.core.kernels[0].shape[0] == d:
                self.core.kernels[0] = orig_kernels[0] + 0.08 * E

            for _ in range(2):
                self.core.couple()

            f = self._fitness()
            fitnesses.append(f)

            # --- Restore full trial-mutated state ---
            self.core.coherent_state = orig_coherent.copy()
            self.core.coherent_phase = orig_phase.copy()
            self.core.void_state = orig_void.copy()
            self.core.void_phase = orig_void_phase.copy()
            self.core.seq_buffer = orig_seq.copy()
            self.core.kernels = [k.copy() for k in orig_kernels]
            self.core.steps = orig_steps

        fitnesses = np.array(fitnesses)

        # --- FIXED rank-weight polarity ---
        # rank 0 = best. We want best → positive weight.
        ranks = np.argsort(np.argsort(-fitnesses))          # 0 = best
        shaped = -((ranks.astype(float) / max(1, self.pop_size - 1) - 0.5) * 2.0)
        # now best ≈ +1, worst ≈ -1

        update = np.zeros((d, d))
        for E, w in zip(residuals, shaped):
            update += w * E
        update /= (self.pop_size * max(1e-6, self.sigma))

        lr_scale = self.update_drift_controller()
        current_lock = self.core.phase_lock_metric(self.core.coherent_phase)
        adaptive_lr = self.base_lr * lr_scale * (0.50 + 0.50 * current_lock)

        state_delta = np.mean(update, axis=1) * adaptive_lr
        self.core.coherent_state += state_delta

        if self.smoothed_drift is not None and self.smoothed_drift > 0.08:
            mean_c = np.mean(self.core.coherent_state)
            damp = 0.04 + 0.06 * min(1.0, self.smoothed_drift)
            self.core.coherent_state = (1.0 - damp) * self.core.coherent_state + damp * mean_c

        for ki in range(min(3, len(self.core.kernels))):
            k = self.core.kernels[ki]
            if k.shape[0] == d and k.shape[1] == d:
                self.core.kernels[ki] = k + 0.10 * adaptive_lr * update

        self.core._enforce_fraction(hard=False)

        for _ in range(n_core_steps):
            self.core.step()

        fit = float(np.mean(fitnesses))
        lock = self.core.phase_lock_metric(self.core.coherent_phase)
        core_l, bridge_l, edge_l = self.core.hierarchical_locks()
        frac = self.core.coherent_fraction()
        drift_now = self._current_drift()

        if fit > self.best_fitness:
            self.best_fitness = fit
            self.best_lock = lock
            self.best_coherent = self.core.coherent_state.copy()

        rec = {
            "gen": self.generation,
            "fitness": round(fit, 5),
            "lock": round(lock, 5),
            "core_l": round(core_l, 4),
            "bridge_l": round(bridge_l, 4),
            "edge_l": round(edge_l, 4),
            "frac": round(frac, 5),
            "drift": round(drift_now, 5),
            "lr_scale": round(self.lr_scale, 4),
            "sigma": round(self.sigma, 5),
            "calm": self.calm_streak,
            "locked": self.locked,
            "shaped_best": float(shaped[np.argmin(ranks)]),   # should now be ≈ +1
            "shaped_worst": float(shaped[np.argmax(ranks)]),  # should now be ≈ -1
        }
        self.history.append(rec)
        self.generation += 1
        return rec