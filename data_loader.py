"""
Data loading and validation for the MM Disease Model Explorer.
"""
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Column Registry ──────────────────────────────────────────────
TOTAL_COLS   = ['Total_1L', 'Total_2L', 'Total_3L', 'Total_4L+']
LINE_LABELS  = {'Total_1L': '1st Line', 'Total_2L': '2nd Line',
                'Total_3L': '3rd Line', 'Total_4L+': '4th Line+'}
LINE_ORDER   = ['1L', '2L', '3L', '4L+']

def _resolve(base: Path, *candidates) -> Path:
    for c in candidates:
        p = base / c
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} found under {base}")


def load_simulation(path: Path) -> pd.DataFrame:
    """Load the detailed (regimen-level) simulation output."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    # Ensure totals exist
    for c in TOTAL_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df.fillna(0)
    return df


def load_legacy(path: Path) -> pd.DataFrame:
    """Load the legacy mm_lot_monthly.csv (simpler schema)."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def load_params(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_regimens_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_regimen_columns(df: pd.DataFrame, line: str) -> list:
    """Return columns for a specific line prefix like '1L_', '2L_', etc."""
    prefix = f"{line}_"
    return [c for c in df.columns if c.startswith(prefix) and not c.startswith('Total_')]


def auto_discover_paths(base: Path) -> Dict[str, Path]:
    """Automatically discover all required data paths."""
    paths = {}

    # Simulation output
    for cand in ['outputs/mm_detailed_simulation.csv',
                 '../outputs/mm_detailed_simulation.csv']:
        p = base / cand
        if p.exists():
            paths['simulation'] = p
            break

    # Legacy output
    for cand in ['outputs/mm_lot_monthly.csv',
                 '../Myeloma/outputs/mm_lot_monthly.csv']:
        p = base / cand
        if p.exists():
            paths['legacy'] = p
            break

    # Params
    for cand in ['params.yaml', '../params.yaml']:
        p = base / cand
        if p.exists():
            paths['params'] = p
            break

    # Regimens
    for cand in ['../regimens.yaml', 'regimens.yaml']:
        p = base / cand
        if p.exists():
            paths['regimens'] = p
            break

    return paths
