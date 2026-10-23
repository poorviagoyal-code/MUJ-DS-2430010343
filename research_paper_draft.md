# Transformer-Based Forecasting of Safely Managed Drinking-Water Access Using Socioeconomic Indicators

**Research manuscript draft — results and references require author verification before submission**

**Author:** Poorvi Goyal  
**Affiliation:** Manipal University Jaipur  
**Project repository:** https://github.com/poorviagoyal-code/MUJ-DS-2430010343

## Abstract

Reliable monitoring of safely managed drinking-water access is important for public-health planning, yet country-level indicators are observed at irregular intervals and may change slowly over time. This study evaluates whether a small Transformer encoder can forecast the next year's drinking-water access from five-year histories of water access, urban population, and unemployment. Country-level WHO/UNICEF Joint Monitoring Programme (JMP) water-access observations are merged with World Bank indicators. A chronological split assigns target years through 2018 to training, 2019–2020 to validation, and 2021–2024 to testing. Linear Regression, Random Forest, a Transformer encoder, and a persistence baseline (repeat the most recent observed water-access value) are compared using mean absolute error (MAE), root mean squared error (RMSE), and coefficient of determination (R²). In the currently saved results, Linear Regression has the lowest MAE (0.0867) and RMSE (0.2446), while the Transformer records MAE 0.4341 and RMSE 0.6169. The persistence baseline (MAE 0.3220; RMSE 0.5778) outperforms both the Transformer and Random Forest. These preliminary results do not support a claim that the Transformer improves forecasting accuracy. Instead, they highlight the importance of simple baselines, careful temporal evaluation, and stronger feature coverage before adopting a more complex architecture. Metrics are reported in the target's native units; the target scale and evaluation pipeline must be verified against the raw source data before publication.

**Keywords:** drinking water; time-series forecasting; Transformer; public health; socioeconomic indicators; baseline comparison

## 1. Introduction

Safely managed drinking-water services are a core component of public health and sustainable development. Monitoring changes in access can help researchers and planners identify where progress is slow and where additional data collection or intervention may be warranted. However, country-level indicators are not always observed annually, and forecasts must therefore be interpreted cautiously.

Classical regression methods can be competitive when data are limited and the target changes gradually. Tree-based models can represent nonlinear relationships, while Transformer architectures use attention mechanisms to model relationships within sequences. A more complex model is not automatically more accurate, particularly when the training sample is small or the recent target value is already a strong predictor.

This project investigates the following question: **Does a Transformer encoder improve one-year-ahead forecasts of safely managed drinking-water access compared with simpler regression models and a persistence baseline, using a five-year history of water access and two socioeconomic indicators?**

The study makes three practical contributions: (1) an integrated country-year dataset using water-access, urban-population, and unemployment indicators; (2) a chronological model comparison with training-only scaling; and (3) an explicit persistence baseline to test whether model complexity adds predictive value.

## 2. Data

### 2.1 Sources and variables

The repository contains three CSV files:

- `water.csv`: water-access observations, filtered to country-level records and the “Total” residence category.
- `urban_population.csv`: World Bank urban population as a percentage of total population.
- `unemployment.csv`: World Bank unemployment indicator.

Records are joined using country code and year. Regional aggregates are removed from the World Bank tables. The target is the numeric water-access value supplied by the source. The current project README reports 3,873 merged country-year rows across 158 countries over 2000–2024, with approximately 5.99% missing urban-population values and 15.00% missing unemployment values. These dataset counts should be regenerated and confirmed from the exact files used for the final run.

### 2.2 Sample construction

For each country, observations are sorted chronologically. A sample is created only when six consecutive annual records are available: five years of input features and the following year's water-access target. Rows with missing feature values are omitted from a window. This prevents a window from silently spanning missing calendar years, but it can also reduce the number of usable samples and may introduce selection bias if missingness is systematic.

The input sequence contains three variables per year: water access, urban population, and unemployment. The model predicts the water-access value for the next year.

### 2.3 Data-quality checks required before submission

The final experiment should explicitly report: raw and merged row counts; unique countries; target units and range; duplicate country-year checks; missingness before and after filtering; and the number of windows in each split. Because the current error metrics are small relative to a typical 0–100 percentage scale, confirm whether the water-access target is stored as a percentage or a fraction. Do not relabel or convert the values without verifying the source metadata.

## 3. Methods

### 3.1 Forecasting setup

Let each input window be (X_t \in \mathbb{R}^{5\times3}). The target is the water-access observation at the next time step. The five-year window length is fixed for this experiment.

### 3.2 Chronological evaluation

The existing pipeline uses target year to form a chronological split:

- Training: target years up to and including 2018.
- Validation: target years 2019–2020.
- Test: target years 2021–2024.

No random shuffling is used to assign samples to these time periods. Standardization parameters are fitted on the training subset only and then applied to validation and test data. This reduces leakage from future distributions. Since windows for a country may share historical observations, results should be interpreted as rolling one-year-ahead forecasting rather than independent-country generalization.

### 3.3 Models

**Linear Regression.** A linear model is trained on the flattened five-year feature window. It provides a simple low-complexity reference.

**Random Forest.** A Random Forest regressor with 100 trees and maximum depth 12 is fitted to the same flattened windows.

**Transformer encoder.** Each yearly feature vector is projected to 32 dimensions. A learned positional embedding is added, followed by two Transformer encoder layers with four attention heads, a feed-forward dimension of 64, and dropout of 0.1. The final time-step representation is passed through layer normalization and a linear output layer. The model is trained using AdamW, mean squared error, gradient clipping, early stopping based on validation loss, and a fixed seed of 42.

**Persistence baseline.** The next year's value is predicted as the most recent water-access observation in the input window. This baseline is essential for a slowly changing indicator.

### 3.4 Metrics

The study reports mean absolute error (MAE), root mean squared error (RMSE), and R² on the held-out test set. MAE and RMSE measure prediction error in the target's native units; RMSE penalizes larger errors more heavily. R² is reported for completeness but is not used alone to choose a model. Forecasting performance is primarily interpreted relative to the persistence baseline.

## 4. Results

The repository's currently saved `model_comparison.csv` reports the following results for test target years 2021–2024:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.0867 | 0.2446 | 0.99993 |
| Random Forest | 0.4012 | 0.6368 | 0.99953 |
| Transformer | 0.4341 | 0.6169 | 0.99956 |
| Persistence (repeat last year) | 0.3220 | 0.5778 | 0.99961 |

*Table 1. Preliminary saved test metrics. Values must be regenerated from the final reproducible run before submission.*

Linear Regression has the lowest saved MAE and RMSE. The persistence baseline performs better than both Random Forest and Transformer on MAE and RMSE. Therefore, the current evidence does not demonstrate that the Transformer adds predictive value over simpler alternatives. The very high R² values should not be interpreted as proof of useful forecasting performance without considering the scale, temporal persistence, and error metrics.

Before this table is treated as final, rerun the pipeline from a clean environment and verify that the reported sample count, split masks, metric units, prediction file, and model-comparison table all correspond to the same run. In particular, confirm the target's units and investigate why the saved linear model's error is substantially smaller than the persistence baseline.

## 5. Discussion

The results illustrate why a neural architecture should be evaluated against both conventional models and a simple persistence forecast. Water-access indicators may evolve gradually, so the latest observation can be a strong predictor of the next one. When a persistence baseline performs well, a complex model must demonstrate a clear and reproducible improvement to justify its additional complexity.

The Linear Regression result is promising in the saved metrics but requires verification. Possible explanations for unexpectedly low error include the target's numeric scale, data construction, the relationship between lagged target values and the next observation, and implementation or evaluation inconsistencies. These possibilities should be investigated rather than assumed to be evidence of generalizable accuracy.

The current feature set is limited to water access, urban population, and unemployment. It omits potentially relevant factors such as infrastructure investment, rainfall and drought, water quality, income inequality, conflict, population growth, and service reliability. Country-level aggregates can also hide regional and socioeconomic inequalities. The analysis is predictive and observational; it does not establish causal effects of unemployment or urbanization on water access.

## 6. Limitations and future work

1. **Target scale and metadata:** verify whether values are fractions or percentages and document any transformations.
2. **Data availability:** missing indicators reduce usable windows and may bias which countries are represented.
3. **Limited predictors:** three variables cannot capture all drivers of water-service access.
4. **Temporal coverage:** a short annual history and relatively small training set may be insufficient for a Transformer.
5. **Validation design:** add rolling-origin evaluation and report metrics separately by target year and country where sample sizes permit.
6. **Uncertainty:** provide prediction intervals or bootstrap uncertainty estimates rather than only point forecasts.
7. **Robustness:** repeat training across multiple seeds and report mean and standard deviation for the Transformer.
8. **Comparability:** use identical windows, target transformations, and split definitions for all models.
9. **External validity:** evaluate performance on countries or time periods not used during model development before claiming broad generalization.

Future experiments should test persistence, regularized linear regression, and carefully tuned tree-based models first. A Transformer should be retained as the preferred method only if robust, leakage-aware tests show a meaningful improvement over these baselines.

## 7. Conclusion

This project develops a country-level, one-year-ahead forecasting pipeline for safely managed drinking-water access using five-year sequences of water access, urban population, and unemployment. In the currently saved results, Linear Regression has the lowest error, and the persistence baseline outperforms both Random Forest and Transformer. The present evidence therefore does not support claiming that the Transformer is the best model. The most important next steps are to verify the target scale, rerun the experiment reproducibly, audit the unexpectedly low linear-regression error, and broaden the temporal and feature-based evaluation.

## References

1. World Health Organization (WHO) and United Nations Children's Fund (UNICEF), Joint Monitoring Programme for Water Supply, Sanitation and Hygiene (JMP). *Progress on household drinking water, sanitation and hygiene*. Indicator definitions and country estimates: https://washdata.org/
2. World Bank. *World Development Indicators*. Urban population and unemployment indicators: https://databank.worldbank.org/source/world-development-indicators
3. Vaswani, A., et al. (2017). “Attention Is All You Need.” *Advances in Neural Information Processing Systems*. https://arxiv.org/abs/1706.03762
4. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*. OTexts. https://otexts.com/fpp3/

## Reproducibility statement

The implementation and input files are maintained in the project repository: https://github.com/poorviagoyal-code/MUJ-DS-2430010343. The manuscript is a draft based on the currently committed result table, not a claim of peer-reviewed validation. Before submission, record the run date, Python and package versions, exact dataset source URLs and retrieval dates, commit hash, final sample counts, and regenerated outputs. Cite and comply with the data providers' terms. Do not submit this draft as a final paper until the checks described above have been completed.
