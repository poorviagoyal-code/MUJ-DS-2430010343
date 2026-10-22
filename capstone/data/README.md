# Input data

Expected files:

- `water.csv` — WHO Global Health Observatory water-access observations
- `urban_population.csv` — World Bank urban-population indicator export
- `unemployment.csv` — World Bank unemployment indicator export

Place the three CSVs in this directory before running the forecasting script. The World Bank exports are expected to have four metadata rows before their header. Record the original source URLs, indicator codes, access dates, and any licensing notes in this README before final submission. The target indicator/category must be verified independently; `Dim1 == Total` alone does not establish that it is the intended safely managed drinking-water series.
