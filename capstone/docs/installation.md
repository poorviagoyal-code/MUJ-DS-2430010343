# Installation and reproducibility

Requires Python 3.9 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python code/water_forecast.py
```

The script expects `water.csv`, `urban_population.csv`, and `unemployment.csv` inside `capstone/data/`. The World Bank CSV files are read after skipping four metadata rows. Confirm the source data layout if the download format changes. Outputs are written under `capstone/results/`, `capstone/models/`, and `capstone/screenshots/`.

For interactive work, open `notebooks/deep_learning_final.ipynb` in Jupyter or Colab. Run all cells from a fresh kernel and check that the displayed results and saved artifacts correspond to the same run.
