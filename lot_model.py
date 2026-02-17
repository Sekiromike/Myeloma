"""
Line-of-Therapy (LoT) Patient Stock Model for US Multiple Myeloma.
Updated to support Regimen-Specific Cohorts, Parametric Survival, and Dynamic Adoption.
Includes Incidence Projection to 2026.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import List, Dict

# Local imports
try:
    from Myeloma.scientific_utils import load_regimens, Regimen, WeibullParams
    from Myeloma.adoption import AdoptionEngine
except ImportError:
    from scientific_utils import load_regimens, Regimen, WeibullParams
    from adoption import AdoptionEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Cohort:
    id: str
    entry_date: pd.Timestamp
    line: str # 1L, 2L, 3L, 4L+
    eligibility: str
    regimen: Regimen
    initial_size: float
    current_size: float
    
    def update(self, current_date: pd.Timestamp, mortality_rate: float) -> tuple[float, float]:
        if self.current_size <= 1e-6:
            return 0.0, 0.0
            
        mo_diff = (current_date.year - self.entry_date.year) * 12 + (current_date.month - self.entry_date.month)
        
        prob_prog = self.regimen.weibull.monthly_transition_prob(float(mo_diff))
        prob_death = mortality_rate 
        
        total_prob = prob_prog + prob_death
        if total_prob > 1.0:
            scale = 1.0 / total_prob
            prob_prog *= scale
            prob_death *= scale
            
        n_prog = self.current_size * prob_prog
        n_death = self.current_size * prob_death
        
        self.current_size = max(0, self.current_size - (n_prog + n_death))
        return n_prog, n_death

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(df_inc, params, regimens, events):
    logger.info("Initializing Simulation...")
    
    adoption_engine = AdoptionEngine(regimens, events['events'])
    
    start_date = df_inc['Date'].min()
    end_date = df_inc['Date'].max()
    dates = pd.date_range(start_date, end_date, freq='MS')
    
    cohorts: List[Cohort] = []
    results_list = []
    
    frac_te = 0.4 
    delay_months = int(params['uptake'].get('dx_to_1l_delay_months', 1))
    p_reach_2l = params['attrition']['p_reach_2l']
    p_reach_3l = params['attrition'].get('p_reach_3l_given_2l', 0.6)
    p_reach_4l = 0.5 
    
    inc_map = df_inc.set_index('Date')['Combined_Incidence'].to_dict()
    
    logger.info(f"Simulating {len(dates)} months from {start_date.date()} to {end_date.date()}...")
    
    for i, current_date in enumerate(dates):
        # 1. NEW 1L STARTS
        diagnosis_date = current_date - pd.DateOffset(months=delay_months)
        new_cases = inc_map.get(diagnosis_date, 0)
        
        treated_fraction = params['uptake']['treated_fraction']
        n_started = new_cases * treated_fraction
        
        n_te = n_started * frac_te
        n_ti = n_started * (1 - frac_te)
        
        shares_1l_te = adoption_engine.get_market_share(current_date, "1L", "TE")
        shares_1l_ti = adoption_engine.get_market_share(current_date, "1L", "TI")
        
        new_cohorts_this_step = []

        def create_cohorts(n_total, shares, line, elig):
            for r_name, share in shares.items():
                if share > 0 and r_name in regimens:
                    size = n_total * share
                    if size > 1e-4:
                        new_cohorts_this_step.append(Cohort(
                            id=f"{current_date.date()}_{line}_{elig}_{r_name}",
                            entry_date=current_date,
                            line=line,
                            eligibility=elig,
                            regimen=regimens[r_name],
                            initial_size=size,
                            current_size=size
                        ))
        
        create_cohorts(n_te, shares_1l_te, "1L", "TE")
        create_cohorts(n_ti, shares_1l_ti, "1L", "TI")
        
        # 2. UPDATE EXISTING COHORTS
        pool_2l_needed = 0
        pool_3l_needed = 0
        pool_4l_needed = 0
        
        monthly_stats = {'Date': current_date, 'New_Starts_1L': n_started}
        
        cohorts.extend(new_cohorts_this_step)
        
        for c in cohorts:
            if c.current_size <= 1e-6:
                continue
                
            if "1L" in c.line: m_rate = params['mortality']['monthly_death_hazard_1l']
            elif "2L" in c.line: m_rate = params['mortality']['monthly_death_hazard_2l']
            else: m_rate = params['mortality']['monthly_death_hazard_3l_plus']
            
            n_prog, n_death = c.update(current_date, m_rate)
            
            if "1L" in c.line: pool_2l_needed += n_prog
            elif "2L" in c.line: pool_3l_needed += n_prog
            elif "3L" in c.line: pool_4l_needed += n_prog
            
            k = f"{c.line}_{c.regimen.name}"
            monthly_stats[k] = monthly_stats.get(k, 0) + c.current_size
            lk = f"Total_{c.line}"
            monthly_stats[lk] = monthly_stats.get(lk, 0) + c.current_size

        # 3. GENERATE NEXT LINES
        n_start_2l = pool_2l_needed * p_reach_2l
        if n_start_2l > 0.1:
            shares = adoption_engine.get_market_share(current_date, "2L", "Both")
            for r_name, share in shares.items():
                if share > 0 and r_name in regimens:
                    size = n_start_2l * share
                    cohorts.append(Cohort(
                        id=f"{current_date.date()}_2L_Both_{r_name}",
                        entry_date=current_date,
                        line="2L",
                        eligibility="Both",
                        regimen=regimens[r_name],
                        initial_size=size,
                        current_size=size
                    ))

        n_start_3l = pool_3l_needed * p_reach_3l
        if n_start_3l > 0.1:
            shares = adoption_engine.get_market_share(current_date, "3L", "Both")
            for r_name, share in shares.items():
                if share > 0 and r_name in regimens:
                    size = n_start_3l * share
                    cohorts.append(Cohort(
                        id=f"{current_date.date()}_3L_Both_{r_name}",
                        entry_date=current_date,
                        line="3L",
                        eligibility="Both",
                        regimen=regimens[r_name],
                        initial_size=size,
                        current_size=size
                    ))
                    
        n_start_4l = pool_4l_needed * p_reach_4l
        if n_start_4l > 0.1:
            shares = adoption_engine.get_market_share(current_date, "4L+", "Both")
            for r_name, share in shares.items():
                if share > 0 and r_name in regimens:
                    size = n_start_4l * share
                    cohorts.append(Cohort(
                        id=f"{current_date.date()}_4L+_Both_{r_name}",
                        entry_date=current_date,
                        line="4L+",
                        eligibility="Both",
                        regimen=regimens[r_name],
                        initial_size=size,
                        current_size=size
                    ))
        
        if i % 12 == 0:
            cohorts = [c for c in cohorts if c.current_size > 1e-4]
            
        results_list.append(monthly_stats)

    df_res = pd.DataFrame(results_list).fillna(0)
    return df_res

def main():
    base_dir = Path(__file__).resolve().parent
    
    if (base_dir / "params.yaml").exists():
        params_path = base_dir / "params.yaml"
    else:
        params_path = base_dir.parent / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found in {base_dir} or parent")

    root_dir = base_dir.parent 
    regimens_path = root_dir / "regimens.yaml"
    events_path = root_dir / "events.yaml"
    
    if not regimens_path.exists():
        regimens_path = base_dir / "regimens.yaml"
        if not regimens_path.exists():
             raise FileNotFoundError(f"regimens.yaml not found")
             
    if not events_path.exists():
         events_path = base_dir / "events.yaml"

    inc_file = root_dir / "outputs/uscs_myeloma_incidence_monthly.csv"
    if not inc_file.exists():
        inc_file = root_dir / "Myeloma/outputs/uscs_myeloma_incidence_monthly.csv"
    if not inc_file.exists():
        search = list(root_dir.glob("**/uscs_myeloma_incidence_monthly.csv"))
        if search: inc_file = search[0]
        
    if not inc_file.exists():
        logger.error("Incidence file not found.")
        return

    logger.info(f"Loading config from {root_dir}")
    params = load_config(params_path)
    regimens_data = load_config(regimens_path)
    events_data = load_config(events_path)
    
    regimens = load_regimens(regimens_data)
    
    df_inc = pd.read_csv(inc_file)
    
    # Robust Date Parsing
    if 'Date' in df_inc.columns:
        df_inc['Date'] = pd.to_datetime(df_inc['Date'])
    elif 'Year' in df_inc.columns and 'Month' in df_inc.columns:
        df_inc['Date'] = pd.to_datetime(df_inc[['Year', 'Month']].assign(DAY=1))
    else:
         df_inc['Date'] = pd.to_datetime(df_inc.iloc[:, 0])
             
    cols = df_inc.columns
    target = 'Monthly_Cases' if 'Monthly_Cases' in cols else ('Count' if 'Count' in cols else cols[-1])
    
    if target in df_inc.columns:
        df_agg = df_inc.groupby('Date')[target].sum().reset_index()
        df_agg.rename(columns={target: 'Combined_Incidence'}, inplace=True)
    else:
        df_agg = df_inc.copy()
        df_agg.rename(columns={df_agg.columns[1]: 'Combined_Incidence'}, inplace=True)

    # --- PROJECTION LOGIC ---
    max_date = df_agg['Date'].max()
    target_end_year = 2026
    
    if max_date.year < target_end_year:
        logger.info(f"Extending incidence from {max_date.date()} to {target_end_year}-12-31")
        last_year_data = df_agg[df_agg['Date'].dt.year == max_date.year].copy()
        
        projected_frames = [df_agg]
        years_to_add = range(max_date.year + 1, target_end_year + 1)
        
        for yr in years_to_add:
            df_new = last_year_data.copy()
            df_new['Date'] = df_new['Date'] + pd.DateOffset(years=(yr - max_date.year))
            projected_frames.append(df_new)
            
        df_agg = pd.concat(projected_frames).sort_values('Date').reset_index(drop=True)
        # Handle Potential Duplicates if logic imperfect (e.g. partial year)
        df_agg = df_agg.drop_duplicates(subset='Date', keep='first')
        
    results = run_simulation(df_agg, params, regimens, events_data)
    
    out_dir = root_dir / "outputs"
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_dir / "mm_detailed_simulation.csv", index=False)
    logger.info("Simulation Complete. Results saved.")

if __name__ == "__main__":
    main()
