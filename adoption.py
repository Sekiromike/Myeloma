"""
Regimen adoption (market-share) engine.

Each regimen's attractiveness follows a logistic diffusion curve from its uptake
start (FDA approval or guideline listing), optionally multiplied by a logistic
decline once a successor displaces it. Raw scores are normalised across the
regimens competing within a line / eligibility segment, so shares sum to 1.

    raw(t)  = peak / (1 + exp(-speed * (t - uptake - inflection)))     t >= uptake
    decl(t) = floor + (1 - floor) / (1 + exp(speed_d * (t - decline_start)))
    share   = raw * decl / sum(raw * decl)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from scientific_utils import Regimen, decimal_year
except ImportError:  # imported as package module
    from .scientific_utils import Regimen, decimal_year


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class AdoptionEngine:
    def __init__(self, regimens: Dict[str, Regimen], events: List[dict] | None = None,
                 new_launch_speed_multiplier: float = 1.0,
                 new_launch_cutoff: float = 2024.0):
        self.regimens = regimens
        self.events = events or []
        self.speed_mult = new_launch_speed_multiplier
        self.cutoff = new_launch_cutoff

    def candidates(self, line: str, eligibility: str) -> List[Regimen]:
        return [r for r in self.regimens.values()
                if r.line == line and (r.eligibility == 'Both' or eligibility == 'Both'
                                       or r.eligibility == eligibility)]

    def raw_scores(self, r: Regimen, years: np.ndarray) -> np.ndarray:
        a = r.adoption_params
        peak = float(a.get('peak_share', 0.5))
        speed = float(a.get('speed', 1.0))
        if r.uptake_year >= self.cutoff:
            speed *= self.speed_mult
        inflection = float(a.get('inflection_years', a.get('time_to_peak', 2.0)))
        since = years - r.uptake_year
        raw = np.where(since >= 0, peak * _logistic(speed * (since - inflection)), 0.0)

        if 'decline_start' in a:
            start = decimal_year(a['decline_start'])
            floor = float(a.get('floor', 0.0))
            d_speed = float(a.get('decline_speed', 1.0))
            raw = raw * (floor + (1 - floor) * (1 - _logistic(d_speed * (years - start))))
        if r.withdrawn:
            raw = np.where(years >= decimal_year(r.withdrawn), 0.0, raw)
        return raw

    def share_matrix(self, years: np.ndarray, line: str,
                     eligibility: str) -> Tuple[List[str], np.ndarray]:
        """Return (regimen keys, shares[T, R]) for a segment over a time grid."""
        cands = self.candidates(line, eligibility)
        if not cands:
            return [], np.zeros((len(years), 0))
        raw = np.column_stack([self.raw_scores(r, years) for r in cands])
        total = raw.sum(axis=1, keepdims=True)
        # Before any modelled regimen exists, assign to the earliest-available one
        empty = total[:, 0] <= 1e-12
        if empty.any():
            first = int(np.argmin([r.uptake_year for r in cands]))
            raw[empty, first] = 1.0
            total = raw.sum(axis=1, keepdims=True)
        return [r.key for r in cands], raw / total

    def get_market_share(self, date, line: str, eligibility: str) -> Dict[str, float]:
        """Shares at a single date (kept for backwards compatibility)."""
        year = np.array([decimal_year(date.date() if hasattr(date, 'date') else date)])
        keys, m = self.share_matrix(year, line, eligibility)
        return {self.regimens[k].name: float(m[0, i]) for i, k in enumerate(keys)}
