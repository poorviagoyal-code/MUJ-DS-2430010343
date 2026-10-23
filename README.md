# Forecasting Access to Safely Managed Drinking Water

**Deep Learning Capstone | Research manuscript in progress**

This project studies one-year-ahead forecasts of safely managed drinking-water access using a five-year history of water access, urban population, and unemployment. It compares Linear Regression, Random Forest, a Transformer encoder, and a persistence baseline.

## Research question

Does a Transformer encoder improve one-year-ahead country-level forecasts over simpler regression methods and repeating the most recent observed water-access value?

## Data

The current raw input files are in the repository root:

| File | Source / purpose |
|---|---|
| `water.csv` | WHO/UNICEF JMP-related water-access indicator data; target and lag feature |
| `urban_population.csv` | World Bank urban population indicator |
| `unemployment.csv` | World Bank unemployment indicator |

The current pipeline filters water-access records to country-level observations with the “Total” residence category, merges by country code and year, and creates windows requiring consecutive annual observations. Check the source metadata and target units before interpreting the metrics.

## Method

- Input: five consecutive years of water access, urban population, and unemployment.
- Target: water-access value in the following year.
- Chronological split by target year: training through 2018, validation 2019–2020, test 2021–2024.
- Standardization parameters are fitted on training data only.
- Models: Linear Regression, Random Forest, Transformer encoder, and persistence (repeat last observed value).

## Preliminary saved results

These are the values currently recorded in `model_comparison.csv`. They must be regenerated and checked from a clean run before being used as final research results.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.0867 | 0.2446 | 0.99993 |
| Random Forest | 0.4012 | 0.6368 | 0.99953 |
| Transformer | 0.4341 | 0.6169 | 0.99956 |
| Persistence (repeat last year) | 0.3220 | 0.5778 | 0.99961 |

**Interpretation:** Linear Regression has the lowest saved MAE and RMSE. The persistence baseline outperforms both Random Forest and Transformer on MAE and RMSE. The current results therefore do not support a claim that the Transformer is the best model. The very high R² values should not be interpreted in isolation.

**Important validation task:** verify whether the target is stored as a fraction or a percentage and investigate the unusually low Linear Regression error relative to persistence. Do not silently alter or clip target values to improve metrics.

## Run the pipeline

Install dependencies and run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python water_forecast.py --data-dir . --out-dir results --model-dir models --fig-dir figures
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for validation steps.

## Research manuscript

- [Research paper draft](research_paper_draft.md) — includes abstract, methods, preliminary results, limitations, references, and a reproducibility statement. It is a draft, not a submission-ready or peer-reviewed paper.
- Existing notebook: [deep_learning.ipynb](deep_learning.ipynb)
- Existing presentation: [water_access_forecasting.pptx](water_access_forecasting.pptx)

## Existing artifacts

The repository already contains raw CSVs, `water_forecast.py`, `deep_learning.ipynb`, `model_comparison.csv`, `water_predictions.csv`, `water_transformer.pt`, and plots. These have been retained. The script writes fresh results to the output directories passed as arguments.

## Limitations and next steps

- Confirm target units, indicator metadata, missingness, sample counts, and the exact dataset source/retrieval date.
- Regenerate all metrics and predictions from the same run.
- Investigate the unusually low Linear Regression error.
- Repeat Transformer training across multiple random seeds.
- Add rolling-origin evaluation and error breakdowns by year/country where appropriate.
- Consider additional predictors only when their sources, temporal coverage, and missingness are documented.

## References / data portals

- WHO/UNICEF Joint Monitoring Programme: https://washdata.org/
- World Bank World Development Indicators: https://databank.worldbank.org/source/world-development-indicators
- Vaswani et al., “Attention Is All You Need” (2017): https://arxiv.org/abs/1706.03762
- Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*: https://otexts.com/fpp3/
