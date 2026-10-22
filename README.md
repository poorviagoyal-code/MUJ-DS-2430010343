# Capstone: Forecasting Access to Safely Managed Drinking Water

**T**Members:** [Poorvi Goyal(poorviagoyal-code)], [Bhoomi Kothari(Bhoomii2910)]

## Problem
Given a country's last 5 years of data, predict next year's % of population using safely managed drinking-water services (WHO/UNICEF JMP indicator `WSH_WATER_SAFELY_MANAGED`).

## Data
| File | Source | Used for |
|---|---|---|
| `data/water.csv` | WHO Global Health Observatory | Target and lag feature (country-level, "Total" residence type) |
| `data/urban_population.csv` | World Bank | Urban population (%) feature |
| `data/unemployment.csv` | World Bank | Unemployment (%) feature |

After cleaning and merging: 3,873 country-year rows, 158 countries, 2000-2024. Missing values: urban population 5.99%, unemployment 15.00%; incomplete windows are dropped. Correlation with water access: urban population 0.671, unemployment 0.167.

## Method
1. **Preprocess:** keep country-level totals, reshape World Bank wide tables, merge on country and year, drop aggregates.
2. **Windows:** 5 consecutive years of [water, urban, unemployment] -> next-year water access.
3. **Time-based split (no shuffling):** train <= 2018 (1,839), validation 2019-2020 (266), test 2021-2024 (522). Scalers fit on training data only.
4. **Models:** Linear Regression, Random Forest (100 trees, depth 12), Transformer encoder (2 layers, 4 heads, d_model 32, AdamW, early stopping on validation loss).

## Results (test set, 2021-2024)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| **Linear Regression** | **0.087** | **0.245** | 0.9999 |
| *Naive baseline (repeat last year)* | 0.322 | 0.578 | 0.9996 |
| Random Forest | 0.401 | 0.637 | 0.9995 |
| Transformer | 0.434 | 0.617 | 0.9996 |

Results come from `code/water_forecast.py` (seed 42, test set 2021-2024, 522 predictions). Transformer training is not bit-for-bit reproducible across hardware, so re-running can shift its numbers slightly. The exploratory notebook's printed outputs are from an earlier run; the table above is final.

Screenshots: [`screenshots/`](screenshots/). Predictions: [`results/water_predictions.csv`](results/water_predictions.csv).

## Findings and limitations
- **Linear Regression is the best model** by a wide margin (MAE 0.087 vs 0.322 for the naive baseline).
- **Neither Random Forest nor the Transformer beats the naive "repeat last year" baseline.** The Transformer (MAE 0.434) is no better than the Random Forest (0.401), and both are worse than simply carrying last year's value forward.
- The very high R2 (> 0.999) mostly reflects that water access changes slowly: last year's value already explains almost everything. R2 alone is not a good measure here; compare against the naive baseline.
- The transformer's value is limited on ~1,800 training rows; more data or features would be needed to justify it.
- Only 3 features; no health, income or rainfall data. Missing unemployment data removes about 15% of rows.

## Run it
See [`docs/installation.md`](docs/installation.md).

## Structure
```
capstone/
  data/         raw input CSVs
  results/      model_comparison.csv, water_predictions.csv
  models/       water_transformer.pt (weights + scaler params)
  screenshots/  plots
  docs/         installation.md, issues.md, pr-workflow.md
```
