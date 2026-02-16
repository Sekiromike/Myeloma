"""
Line-of-Therapy (LoT) Patient Stock Model for US Multiple Myeloma.

This model simulates the flow of patients through 1L, 2L, and 3L+ lines of therapy
based on monthly incidence data and mechanistic parameters.

Parameters (placeholders in params.yaml):
- uptake: treated_fraction (share of incidence treated), delays
- attrition: probabilities of reaching next line (p_reach_2l, etc.)
- durations: median duration of therapy (converts to progression hazard)
- mortality: monthly death hazards on treatment

Note: Parameters are currently placeholders and will be calibrated later.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_hazard(median_duration):
    """Converts median duration (months) to monthly hazard rate."""
    if median_duration <= 0:
        return 1.0 # Immediate progression if duration is 0
    return np.log(2) / median_duration

def run_simulation(df, params):
    """
    Simulates patient stocks over time.
    """
    logger.info("Starting simulation...")
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    n_months = len(df)
    dates = df['Date']
    new_dx = df['Combined_Incidence'].values
    
    # Initialize stock arrays
    on_1l = np.zeros(n_months)
    on_2l = np.zeros(n_months)
    on_3l_plus = np.zeros(n_months)
    
    # Unpack params
    treated_fraction = params['uptake']['treated_fraction']
    delay_months = int(params['uptake']['dx_to_1l_delay_months'])
    
    # Attrition probs
    p_reach_2l = params['attrition']['p_reach_2l']
    p_reach_3l = params['attrition']['p_reach_3l_given_2l']
    
    # Hazards (progression)
    h1 = calculate_hazard(params['durations_months_median']['1l'])
    h2 = calculate_hazard(params['durations_months_median']['2l'])
    h3 = calculate_hazard(params['durations_months_median']['3l_plus'])
    
    # Hazards (mortality)
    d1 = params['mortality']['monthly_death_hazard_1l']
    d2 = params['mortality']['monthly_death_hazard_2l']
    d3 = params['mortality']['monthly_death_hazard_3l_plus']
    
    # Simulation Loop
    for t in range(n_months):
        # 1. Inflows
        # Starts 1L
        if t >= delay_months:
            starts_1l = new_dx[t - delay_months] * treated_fraction
        else:
            starts_1l = 0
            
        # 2. Outflows (calculated from current stock)
        # Note: Depending on discretization (start vs end of month), using t-1 for stock is standard.
        # Here we update t based on t-1. So simulate t=0 (initial) separately or handle t-1 access.
        
        if t == 0:
            # Initial state
            # Assuming 0 stocks at t=0 derived from t=0 inflows only
            # Or simpler: Euler integration steps: Stock[t] = Stock[t-1] + In - Out
            
            # For t=0, we define stocks as just the inflows (assuming previous stocks were 0)
            on_1l[t] = max(0, starts_1l)
            on_2l[t] = 0
            on_3l_plus[t] = 0
            continue
            
        # For t > 0, use t-1 stocks
        prev_n1 = on_1l[t-1]
        prev_n2 = on_2l[t-1]
        prev_n3 = on_3l_plus[t-1]
        
        # 1L Flows
        prog_1l = prev_n1 * h1
        death_1l = prev_n1 * d1
        
        # 2L Flows
        starts_2l = prog_1l * p_reach_2l
        prog_2l = prev_n2 * h2
        death_2l = prev_n2 * d2
        
        # 3L+ Flows
        starts_3l = prog_2l * p_reach_3l
        prog_3l = prev_n3 * h3 # Represents failure/terminal event or cycling within 3L+ bucket?
                               # Model spec says "prog3 = N3 * h3", usually implies leaving the stock.
                               # If 3L+ is the final state, where do they go? Death or off-treatment?
                               # We will follow the formula: outflow = prog3 + death3.
        death_3l = prev_n3 * d3
        
        # Update Stocks with guard
        on_1l[t] = max(0, prev_n1 + starts_1l - (prog_1l + death_1l))
        on_2l[t] = max(0, prev_n2 + starts_2l - (prog_2l + death_2l))
        on_3l_plus[t] = max(0, prev_n3 + starts_3l - (prog_3l + death_3l))
        
    # Compile results
    results = pd.DataFrame({
        'Date': dates,
        'NewDx': new_dx,
        'On_1L': on_1l,
        'On_2L': on_2l,
        'On_3L_plus': on_3l_plus
    })
    
    results['On_Treatment_Total'] = results['On_1L'] + results['On_2L'] + results['On_3L_plus']
    
    logger.info("Simulation complete.")
    return results

def main():
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir # Since script is in Myeloma/, parent is correct? Ensure params.yaml location.
    
    # Check if we are in Myeloma/ subdir or root
    if (base_dir / "params.yaml").exists():
        params_path = base_dir / "params.yaml"
        # root is base_dir
    elif (base_dir.parent / "params.yaml").exists():
        params_path = base_dir.parent / "params.yaml"
        base_dir = base_dir.parent # Set base to project root
    else:
        raise FileNotFoundError("params.yaml not found.")
        
    # Input File location
    # Try 'epi outputs' first, then 'outputs'
    incidence_file = base_dir / "Myeloma/epi outputs/uscs_myeloma_incidence_monthly.csv"
    if not incidence_file.exists():
        incidence_file = base_dir / "Myeloma/outputs/uscs_myeloma_incidence_monthly.csv"
        
    if not incidence_file.exists():
        # Fallback for if script is running from root and path logic differs
        incidence_file = base_dir / "outputs/uscs_myeloma_incidence_monthly.csv"
        
    if not incidence_file.exists():
        raise FileNotFoundError(f"Monthly incidence file not found. Checked 'epi outputs' and 'outputs'.")
        
    # Load inputs
    logger.info(f"Loading parameters from {params_path}")
    params = load_config(params_path)
    
    logger.info(f"Loading incidence from {incidence_file}")
    df_inc = pd.read_csv(incidence_file)
    
    # Pre-process Data
    # Ensure Date column
    if 'Date' not in df_inc.columns:
        if 'Year' in df_inc.columns and 'Month' in df_inc.columns:
            logger.info("Creating Date column from Year/Month...")
            df_inc['Date'] = pd.to_datetime(
                df_inc['Year'].astype(str) + '-' + df_inc['Month'].astype(str) + '-01'
            )
        else:
            raise ValueError("Input data missing 'Date' column and cannot reconstruct from Year/Month.")
            
    df_inc['Date'] = pd.to_datetime(df_inc['Date'])
    
    # Identify Incidence Column
    inc_col = None
    for col in ['Monthly_Cases', 'monthly_cases', 'Count', 'count']:
        if col in df_inc.columns:
            inc_col = col
            break
            
    if not inc_col:
        raise ValueError(f"Could not identify monthly incidence column. Available: {df_inc.columns}")
        
    logger.info(f"Using '{inc_col}' as incidence column.")
    
    # Aggregate to total monthly incidence (removing demographic splits)
    df_agg = df_inc.groupby('Date')[inc_col].sum().reset_index()
    df_agg.rename(columns={inc_col: 'Combined_Incidence'}, inplace=True)
    
    # Run Model
    df_results = run_simulation(df_agg, params)
    
    # Output
    # Create outputs folder relative to script or root? Prompt says "creates outputs/ folder if missing". 
    # Usually implies project root outputs or relative to execution.
    # We will use project_root/outputs
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    out_path = output_dir / "mm_lot_monthly.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"Saved results to {out_path}")
    
    # Validation prints
    print("\n--- Validation ---")
    print(f"Date Range: {df_results['Date'].min().date()} to {df_results['Date'].max().date()}")
    print("Non-negative integrity check:")
    cols = ['On_1L', 'On_2L', 'On_3L_plus']
    if (df_results[cols] < 0).any().any():
        print("FAIL: Negative values detected.")
    else:
        print("PASS: All stocks non-negative.")
        
    print(f"Rows match input: {len(df_results) == len(df_agg)} ({len(df_results)} rows)")
    
    print("\n--- Tail ---")
    print(df_results.tail())

if __name__ == "__main__":
    main()
