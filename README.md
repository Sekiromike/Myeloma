# Multiple Myeloma Disease Model Explorer 🔬

An interactive scientific dashboard for visualizing the MM microsimulation model.

## Quick Start

```bash
# 1. Install dependencies
pip install -r Myeloma/requirements.txt

# 2. Run the dashboard
streamlit run Myeloma/app.py
```

The app will open at `http://localhost:8501`.

## Features

| Tab | Contents |
|-----|----------|
| **Patient View** | Journey Sankey, treatment line occupancy, duration bars, auto-insights, doctor Q&A, disclaimer |
| **Provider View** | Stacked prevalence, transition rates, regimen share evolution, mortality hazards, sensitivity tornado, CSV/PNG export |

## Sidebar Controls
- **Date range** slider to filter the time window
- **Parameter sliders** for treated fraction, P(reach 2L), P(reach 3L)
- All changes reflect immediately in the charts

## Data Sources
- `outputs/mm_detailed_simulation.csv` — primary simulation output (regimen-level, 1999–2026)
- `Myeloma/params.yaml` — model parameters
- `regimens.yaml` — regimen catalog with adoption calibration

## Schema: `mm_detailed_simulation.csv`

| Column | Description |
|--------|-------------|
| `Date` | Month (YYYY-MM-DD) |
| `New_Starts_1L` | New patients starting 1st-line treatment |
| `Total_1L` / `Total_2L` / `Total_3L` / `Total_4L+` | Total patients on each line |
| `1L_VRd`, `1L_D-VRd`, … | Patients on specific regimens within each line |

## Deployment

```bash
# Streamlit Community Cloud
# 1. Push to GitHub
# 2. Connect repo at share.streamlit.io
# 3. Set main file path: Myeloma/app.py
```

## Architecture

```
Myeloma/
  app.py           # Streamlit dashboard
  data_loader.py   # Data loading & validation
  metrics.py       # Derived metrics (occupancy, transitions, Sankey, insights)
  lot_model.py     # Microsimulation engine
  params.yaml      # Model parameters
  scientific_utils.py  # Weibull, Regimen dataclass
  adoption.py      # Logistic adoption engine
```

## License
Research use only.
