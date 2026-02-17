import numpy as np
import pandas as pd
from typing import Dict, List
from .scientific_utils import Regimen

class AdoptionEngine:
    def __init__(self, regimens: Dict[str, Regimen], events: List[dict]):
        self.regimens = regimens
        self.events = events

    def get_market_share(self, date: pd.Timestamp, line: str, eligibility: str) -> Dict[str, float]:
        """
        Returns a dictionary of {regimen_name: share} sums to 1.0.
        Uses calibrated adoption parameters (peak, speed, time_to_peak) if available.
        """
        current_year = date.year + date.month / 12.0
        
        # Filter relevant regimens
        candidates = []
        for r in self.regimens.values():
            target_line = r.line.replace("_PLUS", "+") 
            req_line = line.replace("_PLUS", "+")
            
            line_match = (target_line == req_line)
            if "4L" in req_line and "4L+" in target_line: line_match = True
            
            if not line_match: continue
            if r.eligibility != 'Both' and r.eligibility != eligibility: continue
            candidates.append(r)

        if not candidates: return {}

        # Calculate Raw Shares based on adoption curves
        raw_shares = {}
        
        for r in candidates:
            years_since = current_year - r.approval_year
            if years_since < 0:
                raw_shares[r.name] = 0.0
                continue
                
            # Calibrated Parameters
            peak = r.adoption_params.get('peak_share', 0.5)
            k = r.adoption_params.get('speed', 1.0)
            # time_to_peak usually means time to saturation. 
            # Logistic curve inflection is at t0. 
            # Let's assume t_peak parameter is rough time to reach ~80% of peak?
            # Or use it as inflection offset.
            # Default offset 2.0 years implies peak growth 2 years after approval.
            t_inflection = r.adoption_params.get('time_to_peak', 2.0)
            
            # Logistic Function
            # S(t) = Peak / (1 + exp(-k * (t - t_inflection)))
            val = peak / (1.0 + np.exp(-k * (years_since - t_inflection)))
            
            raw_shares[r.name] = val

        # Normalize to 1.0
        total_score = sum(raw_shares.values())
        if total_score <= 1e-9:
             return {c.name: 1.0/len(candidates) for c in candidates} if candidates else {}
             
        # Normalize
        final_shares = {name: s / total_score for name, s in raw_shares.items()}
        return final_shares
