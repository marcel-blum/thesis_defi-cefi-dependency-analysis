# How Decentralized is DeFi? A Machine Learning & Time Series Econometrics Study of DeFi-CeFi Dependencies Through Total Value Locked Forecasting

Master's thesis investigating the degree to which the Decentralized Finance (DeFi) ecosystem's Total Value Locked (TVL) depends on centralized finance (CeFi) and macro-financial drivers, and how accurately that dependency can be forecast across a range of modeling paradigms.

The thesis can be provided on request.

## Research Focus

- **Dependent variable:** Total Value Locked (TVL) across the DeFi ecosystem
- **Independent variables:** 38 CeFi/macro-financial and on-chain drivers
- **Core question:** How strong is DeFi's structural dependency on CeFi, and which modeling approach forecasts TVL most accurately?

## Methodology

A unified three-stage evaluation framework is applied across all model classes:

1. **Split:** Chronological 80/20 train/test split (no shuffling, to preserve time-series dependence)
2. **Tuning:** Model selection and hyperparameter tuning via 5-fold blocked cross-validation within the training set (contiguous, non-overlapping blocks)
3. **Evaluation:** Rolling-origin, one-step-ahead forecasting on the held-out test set, assessed via RMSE, MAE, and out-of-sample R²

Model forecasts are finally ranked using a **Model Confidence Set (MCS)** test to identify the set of statistically superior models.

## Models

| Category | Models |
|---|---|
| Linear baseline | Multiple Linear Regression |
| Regularized linear | Elastic Net |
| Non-linear | Generalized Additive Model (GAM) |
| Tree-based | Random Forest, XGBoost |
| Kernel-based | Support Vector Machine (RBF kernel) |
| Time series econometrics | SARIMAX / ARX |
| Deep learning | LSTM, Temporal Fusion Transformer (TFT) |

## Repository Structure

- `0_a_helpers/` — shared plotting/table formatting utilities
- `0_data/` — data pipeline (raw source data excluded from this repo due to licensing; processed visualizations/tables included)
- `1_data_exploration_pre-tests/` — EDA, stationarity (ADF), correlation(Pearson/Spearman), multicollinearity (VIF), descriptive statistics
- `2_simple_models/` — MLR baseline
- `3_model_improvement/` — Elastic Net
- `4_non_linear_models/` — GAM
- `5_tree_models/` — Random Forest, XGBoost
- `6_support_vector_machines/` — SVM (RBF kernel)
- `7_sarimax/` — SARIMAX/ARX
- `8_deep_learning_model/` — LSTM, Temporal Fusion Transformer
- `9_Model_Comparison/` — cross-model comparison and Model Confidence Set test

## Data

The underlying dataset is not fully included in this repository due to licensing restrictions on several source series. Generated tables and visualizations are included where possible.

## LaTeX compilation in PyCharm:
RUN "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/strip_alpha_fix.py" AS SOON AS A NEW .png FILE IS CREATED

PDF COMPILATION USING TERMINAL:
cd /Users/marcel/PycharmProjects/Master_Thesis_Pavia/thesis
rm -f *.aux *.bbl *.bcf *.blg *.run.xml *.out *.toc *.lof *.lot *.equ *.log
pdflatex 01_main.tex
biber 01_main
pdflatex 01_main.tex
pdflatex 01_main.tex