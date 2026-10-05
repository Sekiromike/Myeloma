"""
Parametric survival utilities and the regimen catalogue schema.

Each regimen's on-line survival is a Weibull distribution calibrated either to a
published median PFS or, where the median has not been reached, to a published
landmark PFS rate (e.g. PERSEUS 48-month PFS = 84.3%).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

LN2 = np.log(2.0)


def decimal_year(value) -> float:
    """Convert 'YYYY-MM-DD', a date, or a year number to a decimal year."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = dt.date.fromisoformat(value)
    start = dt.date(value.year, 1, 1)
    days = (dt.date(value.year + 1, 1, 1) - start).days
    return value.year + (value - start).days / days


@dataclass
class WeibullParams:
    scale: float  # lambda (months)
    shape: float  # k

    @classmethod
    def from_median(cls, median: float, shape: float = 1.3) -> "WeibullParams":
        """median = scale * ln(2)^(1/shape)."""
        if median is None or not np.isfinite(median) or median <= 0:
            raise ValueError(f"Invalid median: {median}")
        return cls(scale=median / LN2 ** (1.0 / shape), shape=shape)

    @classmethod
    def from_landmark(cls, month: float, survival: float,
                      shape: float = 1.3) -> "WeibullParams":
        """S(t) = exp(-(t/scale)^shape)  =>  scale = t / (-ln S)^(1/shape)."""
        if not (0 < survival < 1) or month <= 0:
            raise ValueError(f"Invalid landmark: S({month}) = {survival}")
        return cls(scale=month / (-np.log(survival)) ** (1.0 / shape), shape=shape)

    @property
    def median(self) -> float:
        return self.scale * LN2 ** (1.0 / self.shape)

    def cumulative_hazard(self, t):
        t = np.clip(np.asarray(t, dtype=float), 0, None)
        return (t / self.scale) ** self.shape

    def survival_prob(self, t):
        return np.exp(-self.cumulative_hazard(t))

    def hazard_rate(self, t):
        t = np.asarray(t, dtype=float)
        return np.where(t > 0, (self.shape / self.scale) * (np.clip(t, 1e-12, None) / self.scale) ** (self.shape - 1), 0.0)

    def monthly_transition_prob(self, t_start: float, dt_: float = 1.0) -> float:
        s0 = float(self.survival_prob(t_start))
        return 1.0 if s0 == 0 else 1.0 - float(self.survival_prob(t_start + dt_)) / s0

    def scaled(self, multiplier: float) -> "WeibullParams":
        """Stretch the time axis (multiplier < 1 = shorter real-world PFS)."""
        return WeibullParams(scale=self.scale * multiplier, shape=self.shape)


@dataclass
class Regimen:
    name: str
    line: str                       # 1L, 2L, 3L, 4L+
    eligibility: str                # TE, TI, Both
    approval_date: str
    citation: str
    label: str = ""
    trial: str = ""
    classes: List[str] = field(default_factory=list)
    pfs_median: Optional[float] = None
    pfs_landmark: Optional[Dict[str, float]] = None
    hazard_ratio: Optional[float] = None
    comparator: str = ""
    uptake_start: Optional[str] = None
    withdrawn: Optional[str] = None
    note: str = ""
    adoption_params: Dict[str, float] = field(default_factory=dict)
    weibull_shape: float = 1.3
    weibull: WeibullParams = field(init=False)

    def __post_init__(self):
        self.label = self.label or self.name
        if self.pfs_landmark:
            self.weibull = WeibullParams.from_landmark(
                self.pfs_landmark['month'], self.pfs_landmark['survival'], self.weibull_shape)
        else:
            self.weibull = WeibullParams.from_median(self.pfs_median, self.weibull_shape)

    @property
    def key(self) -> str:
        """Unique identifier, also the simulation output column name."""
        return f"{self.line}_{self.name}"

    @property
    def approval_year(self) -> float:
        return decimal_year(self.approval_date)

    @property
    def uptake_year(self) -> float:
        return decimal_year(self.uptake_start or self.approval_date)

    @property
    def implied_median(self) -> float:
        return self.weibull.median

    @property
    def efficacy_basis(self) -> str:
        if self.pfs_landmark:
            m, s = self.pfs_landmark['month'], self.pfs_landmark['survival']
            return f"{int(m)}-mo PFS {s:.1%} (median NR)"
        return f"median PFS {self.pfs_median:g} mo"


def load_regimens(data: dict) -> Dict[str, Regimen]:
    """Parse regimens.yaml into Regimen objects keyed by '<line>_<name>'."""
    shape = data.get('defaults', {}).get('weibull_shape', 1.3)
    out: Dict[str, Regimen] = {}
    for r in data.get('regimens', []):
        reg = Regimen(
            name=r['name'],
            label=r.get('label', r['name']),
            line=str(r['line']).upper(),
            eligibility=r.get('eligibility', 'Both'),
            approval_date=str(r['approval_date']),
            uptake_start=r.get('uptake_start'),
            withdrawn=r.get('withdrawn'),
            citation=r.get('citation', ''),
            trial=r.get('trial', ''),
            classes=list(r.get('classes', [])),
            pfs_median=r.get('pfs_median'),
            pfs_landmark=r.get('pfs_landmark'),
            hazard_ratio=r.get('hazard_ratio'),
            comparator=r.get('comparator', ''),
            note=r.get('note', ''),
            adoption_params=r.get('adoption', {}),
            weibull_shape=r.get('weibull_shape', shape),
        )
        if reg.key in out:
            raise ValueError(f"Duplicate regimen key {reg.key}")
        out[reg.key] = reg
    return out
