# Reproducing the water-access forecasting project

Repository: https://github.com/poorviagoyal-code/MUJ-DS-2430010343

## Current layout

The existing data files and Python script are in the repository root. Do not move or delete them just to match an older proposed directory tree.

## Environment

Use Python 3.10–3.12 where possible. Create a virtual environment, activate it, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Run

From the repository root, explicitly provide the current root-level data directory and output folders:

```bash
python water_forecast.py --data-dir . --out-dir results --model-dir models --fig-dir figures
```

The script expects these files in the data directory:

- `water.csv`
- `urban_population.csv`
- `unemployment.csv`

It writes `model_comparison.csv` and `water_predictions.csv` to the results folder, the model checkpoint to models, and the actual-versus-predicted figure to figures. Check the script's console output and confirm that every output comes from the same run.

## Before citing results

1. Verify the target column's source definition and whether its values are percentages (0–100) or proportions (0–1).
2. Record the merged data shape, country count, and train/validation/test sample counts from the actual run.
3. Compare every model against the persistence baseline on exactly the same test rows.
4. Check that predictions remain within a meaningful range; if the target is a percentage, investigate values outside 0–100 rather than silently clipping them.
5. Rerun from a clean environment and retain the generated comparison table and predictions.
6. Record Python, PyTorch, pandas, NumPy, and scikit-learn versions, the repository commit hash, and source-data retrieval dates.

The committed CSV and figures are prior outputs; they are not proof that a fresh run has been reproduced. The manuscript in `research_paper_draft.md` is a draft and should be updated only after these checks.
