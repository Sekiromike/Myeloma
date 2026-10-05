# Myeloma Forecast Studio

US multiple myeloma epidemiology and line-of-therapy forecast (1999–2035), shown in an
interactive Streamlit dashboard. Every sidebar lever re-runs the full simulation live, in about 30 ms.

## Quick start

```bash
pip install -r Myeloma/requirements.txt
streamlit run Myeloma/app.py          # dashboard on http://localhost:8501
python Myeloma/lot_model.py           # regenerate outputs/*.csv (base case)
pytest Myeloma/tests                  # scientific invariant checks
```

## Dashboard

| Tab | Contents |
|-----|----------|
| **Overview** | Patients on therapy by line (observed vs projected), annual snapshot, line starts |
| **Patient flow** | 10-year journey of 1,000 patients starting 1L (Sankey), exits by line, transition rates |
| **Regimen forecast** | Regimen mix by line (patients or new starts), regimen tables, drug-class exposure, CSV export |
| **Epidemiology** | USCS observed vs projected incidence, ACS 2026 benchmark, age/sex-specific rates, ageing effect |
| **Evidence & approvals** | Efficacy inputs (trial vs real-world medians), evidence register with citations, FDA timeline to Dec 2026 |
| **Validation & sensitivity** | External benchmarks (ACS, SEER, real-world attrition), re-simulated tornado, scenario comparison |
| **Methods** | Model diagram, equations, ranked limitations, patient guide |

## Data (as of October 2026)

| Input | Source |
|-------|--------|
| Incidence 1999–2022 by age/sex | CDC WONDER, U.S. Cancer Statistics (NPCR + SEER) |
| Population projections 2022–2045 | U.S. Census Bureau 2023 National Projections, middle series (`data/`) |
| Benchmarks | ACS Cancer Statistics 2026 (36,000 cases, 10,850 deaths); SEER prevalence 2022 (192,144) |
| Regimen efficacy | Pivotal-trial PFS: median, or landmark rate where the median is not reached (`regimens.yaml`) |
| Approvals | FDA, including D-VRd TI (Jan 2026), teclistamab + daratumumab (Mar 2026), BVd (Oct 2025) (`events.yaml`) |

## Architecture

```
Myeloma/
  app.py              Streamlit dashboard (design system, tabs, live re-simulation)
  lot_model.py        Vectorised LoT engine: run_model(), run_cohort(), scenario_from_params()
  epidemiology.py     USCS loader, age-standardised rates, demographic / trend / flat projection
  adoption.py         Logistic diffusion + displacement market-share engine
  scientific_utils.py Weibull (median or landmark calibration), Regimen schema
  model_metrics.py    Annualisation, regimen/class mix, tornado, benchmarks, narrative
  data_loader.py      Legacy CSV helpers (not used by the dashboard)
  params.yaml         Base-case parameters (all exposed as sidebar levers)
  regimens.yaml       Regimen catalogue: efficacy, citations, approvals, adoption calibration
  events.yaml         FDA regulatory timeline
  data/               Census projection extract
  outputs/            Model outputs (mm_detailed_simulation.csv, mm_regimen_starts.csv, ...)
  tests/              pytest invariants (efficacy reproduction, share sums, mass balance)
```

## Output schema: `outputs/mm_detailed_simulation.csv` (monthly)

| Column | Meaning |
|--------|---------|
| `Incidence`, `Untreated`, `New_Starts_1L` | Diagnoses, diagnosed but untreated, 1L starts |
| `<line>_<regimen>` | Patients on that regimen (e.g. `1L_D-VRd-ASCT`, `2L_Tec-Dara`) |
| `Total_1L` … `Total_4L+` | Patients on each line |
| `Starts_<line>`, `Progressions_<line>`, `Deaths_<line>`, `No_Next_Line_<line>` | Monthly flows |

## License
Research use only. Not medical advice.
