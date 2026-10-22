# MUJ-DS-2430010343 — Capstone Repository

**Student:** Poorvi Goyal  
**Registration Number:** 2430010343  
**Branch:** B.Tech Computer Science Engineering (Data Science)  
**Batch:** F (as required by the repository naming convention; confirm with course records)  
**Institution:** Manipal University Jaipur  
**GitHub:** [poorviagoyal-code](https://github.com/poorviagoyal-code)  
**Project:** Forecasting Access to Safely Managed Drinking Water

## Team

- Poorvi Goyal — 2430010343
- Bhoomi Kothari — 2430010325

## Project overview

This project compares Linear Regression, Random Forest, and a Transformer model for one-year-ahead country-level water-access prediction using five-year windows of water access, urban population, and unemployment indicators. The intended evaluation split is chronological: training through 2018, validation during 2019–2020, and test during 2021–2024.

**Research caution:** saved outputs and presentation metrics are provisional until a clean, end-to-end run of the final notebook/script reproduces them. The current water-data filtering uses `Dim1 == Total`; verify the exact WHO indicator/category before claiming it unambiguously represents safely managed drinking-water access. Predictions must be checked to remain within 0–100%. Compare all models against a repeat-last-year baseline on identical test rows.

## Repository structure

```text
.
├── assignments/
├── notebooks/
├── code/
├── resources/
├── presentations/
├── capstone/
│   ├── data/
│   ├── figures/
│   ├── models/
│   ├── results/
│   └── docs/
├── docs/
├── .github/ISSUE_TEMPLATE/
├── .github/PULL_REQUEST_TEMPLATE/
└── requirements.txt
```

## Setup

Use Python 3.9 or later. From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python code/water_forecast.py
```

By default, the script reads the three source CSV files from `capstone/data/`. The World Bank CSVs are expected to have four metadata rows; check the headers before running if the source format changes. Generated metrics and predictions are written to `capstone/results/`, the checkpoint to `capstone/models/`, and the generated plot to `capstone/screenshots/`.

## Reproducibility checklist

- [ ] Verify the exact WHO indicator/service category and the meaning of `Dim1 == Total`.
- [ ] Run the final notebook from a clean kernel or run `python code/water_forecast.py` from the repository root.
- [ ] Confirm the input filenames and data sources.
- [ ] Confirm preprocessing/scalers are fit on training data only.
- [ ] Verify identical test rows for each model and the repeat-last-year baseline.
- [ ] Inspect prediction bounds (0–100%) and regenerate all metrics/figures from that run.
- [ ] Ensure checkpoint, predictions, model comparison, and presentation refer to the same run.
- [ ] Add the instructor as collaborator, and include team members/instructor on the separate team capstone repository per the course instructions.
- [ ] Record meaningful individual commits, issues, and reviewed pull requests.

## Course deliverables

The course guidelines call for a public personal repository named `MUJ-DS-RegNo`, a separate team capstone repository, instructor collaboration on both, source code, documentation, screenshots, results, an installation guide, a presentation, and the URLs submitted for both repositories. A ZIP alone is not the final submission.

See `docs/installation.md`, `docs/issues.md`, `docs/pr-workflow.md`, and `resources/Batch_F_NLP_Capstone_Project_Guidelines_v1_0.pdf`.
