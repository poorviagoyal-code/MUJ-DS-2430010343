# Installation guide

Requires Python 3.9+.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python code/water_forecast.py
```

The script reads files from `capstone/data/` and writes results, models, and generated figures under `capstone/results/`, `capstone/models/`, and `capstone/screenshots/`. It uses a GPU if available; CPU execution is also supported.

Open `notebooks/deep_learning_final.ipynb` for interactive use. Run all cells from a fresh kernel and verify the output artifacts. The World Bank data files are expected to be CSV exports with four metadata rows before the header.
