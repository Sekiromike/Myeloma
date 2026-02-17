import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

@dataclass
class WeibullParams:
    scale: float # lambda
    shape: float # k

    @classmethod
    def from_median(cls, median: float, shape: float = 1.3):
        """
        Derive Weibull parameters from median time-to-event.
        median = scale * (ln(2))^(1/shape)
        => scale = median / (ln(2))^(1/shape)
        """
        if median <= 0 or pd.isna(median):
            return cls(scale=0.1, shape=1.0) 
            
        scale = median / (np.log(2) ** (1 / shape))
        return cls(scale=scale, shape=shape)

    def hazard_rate(self, t: float) -> float:
        if t <= 0: return 0.0
        return (self.shape / self.scale) * ((t / self.scale) ** (self.shape - 1))

    def survival_prob(self, t: float) -> float:
        if t < 0: return 1.0
        return np.exp(-((t / self.scale) ** self.shape))

    def monthly_transition_prob(self, t_start: float, dt: float = 1.0) -> float:
        s_current = self.survival_prob(t_start)
        if s_current == 0:
            return 1.0
        s_next = self.survival_prob(t_start + dt)
        return 1.0 - (s_next / s_current)

@dataclass
class Regimen:
    name: str
    line: str
    eligibility: str # TE, TI, Both
    approval_year: int
    pfs_median: float
    citation: str
    hazard_ratio: Optional[float] = None
    adoption_params: Dict[str, float] = field(default_factory=dict)
    weibull: WeibullParams = field(init=False)
    
    def __post_init__(self):
        # Default shape 1.3 based on oncology literature
        self.weibull = WeibullParams.from_median(self.pfs_median, shape=1.3)

def load_regimens(data: dict) -> Dict[str, Regimen]:
    """Parse YAML data into Regimen objects."""
    regimens = {}
    defaults = data.get('defaults', {})
    default_shape = defaults.get('weibull_shape', 1.3)
    
    for r_data in data.get('regimens', []):
        r = Regimen(
            name=r_data['name'],
            line=str(r_data['line']).upper(),
            eligibility=r_data.get('eligibility', 'Both'),
            approval_year=r_data['approval_year'],
            pfs_median=float(r_data.get('pfs_median', 0)),
            citation=r_data.get('citation', ''),
            hazard_ratio=r_data.get('hazard_ratio'),
            adoption_params=r_data.get('adoption', {})
        )
        r.weibull = WeibullParams.from_median(r.pfs_median, shape=default_shape)
        regimens[r.name] = r
        
    return regimens
