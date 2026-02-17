"""
Data loading and validation for the MM Disease Model Explorer.
Handles both local development and Streamlit Cloud path layouts.
"""
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict
import logging, os

logger = logging.getLogger(__name__)

# ── Column Registry ──────────────────────────────────────────────
TOTAL_COLS   = ['Total_1L', 'Total_2L', 'Total_3L', 'Total_4L+']
LINE_LABELS  = {'Total_1L': '1st Line', 'Total_2L': '2nd Line',
                'Total_3L': '3rd Line', 'Total_4L+': '4th Line+'}
LINE_ORDER   = ['1L', '2L', '3L', '4L+']


def load_simulation(path: Path) -> pd.DataFrame:
    """Load the detailed (regimen-level) simulation output."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    for c in TOTAL_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df.fillna(0)
    return df


def load_legacy(path: Path) -> pd.DataFrame:
    """Load the legacy mm_lot_monthly.csv."""
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


def _find_file(search_dirs: list, candidates: list) -> Path | None:
    """Search multiple directories for the first matching candidate file."""
    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.exists():
                return p
    return None


def auto_discover_paths(base: Path) -> Dict[str, Path]:
    """Discover data paths.  Works for:
       - Local:  base = Myeloma/,  data in Myeloma/ or parent
       - Cloud:  base = /mount/src/myeloma/,  everything at base
    """
    parent = base.parent
    search = [base, parent]                      # local layout
    search += [Path('/mount/src/myeloma')]        # Streamlit Cloud fallback

    paths: Dict[str, Path] = {}

    sim = _find_file(search, [
        Path('outputs/mm_detailed_simulation.csv'),
        Path('outputs') / 'mm_detailed_simulation.csv',
    ])
    if sim:
        paths['simulation'] = sim

    legacy = _find_file(search, [
        Path('outputs/mm_lot_monthly.csv'),
        Path('Myeloma/outputs/mm_lot_monthly.csv'),
    ])
    if legacy:
        paths['legacy'] = legacy

    params = _find_file(search, [Path('params.yaml')])
    if params:
        paths['params'] = params

    regimens = _find_file(search, [Path('regimens.yaml')])
    if regimens:
        paths['regimens'] = regimens

    events = _find_file(search, [Path('events.yaml')])
    if events:
        paths['events'] = events

    return paths
