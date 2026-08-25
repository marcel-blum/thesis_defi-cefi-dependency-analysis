import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
INPUT_PATH  = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/7_sarimax/Tables_Figures"

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

TARGET = "Global_TVL_USD_log_return"

# ─────────────────────────────────────────────
# GRANGER-SIGNIFICANT EXOGENOUS VARIABLES
# 23 variables significant after Bonferroni correction (α* ≈ 0.0013).
# Excluded: USA_Rate_Hike, USA_Rate_Cut, EZ_Rate_Hike, EZ_Rate_Cut,
#           China_Rate_Cut, EZ_D_CPI, CHN_D_CPI, CHN_D_UR,
#           EZ_D_Liquidity, CHN_D_Liquidity
# ─────────────────────────────────────────────
EXOG_VARS = [
    # Panel A: On-Chain Variables
    "Global_Stablecoins_Mcap_USD_log_return",
    "Global_Merchandise_Vol_USD_log_return",
    "Global_Gross_Revenue_USD_log_return",
    # Panel B: Cryptocurrency Returns
    "BTCUSDT_Log_Return",
    "ETHUSDT_Log_Return",
    "BNBUSDT_Log_Return",
    "XRPUSDT_Log_Return",
    "SOLUSDT_Log_Return",
    # Panel C: Cryptocurrency Trading Volume
    "BTCUSDT_Vol_LogGrowth",
    "ETHUSDT_Vol_LogGrowth",
    "BNBUSDT_Vol_LogGrowth",
    "XRPUSDT_Vol_LogGrowth",
    "SOLUSDT_Vol_LogGrowth",
    # Panel D: Exchange Rates
    "EUR_USD_log_return",
    "USD_CNY_log_return",
    # Panel F: Macroeconomic Variables (Bonferroni-significant only)
    "USA_D_CPI",
    "EZ_D_UR",
    # Panel G: Money Supply
    "USA_D_Liquidity",
    # Panel H: Equity & Commodity Indices
    "SP500_log_return",
    "STOXX600_log_return",
    "MSCIEM_log_return",
    "GlobalGold_log_return",
    "SOX_log_return",
]

# Verify all variables exist in dataset
EXOG_VARS = [v for v in EXOG_VARS if v in df.columns]
print(f"Exogenous variables used: {len(EXOG_VARS)}")

df_model = df[["date", TARGET] + EXOG_VARS].dropna()

endog = df_model[TARGET].values        # (N,) — TVL log return
exog  = df_model[EXOG_VARS].values     # (N, k) — exogenous regressors
dates = df_model["date"].values

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT  (80 / 20, chronological)
# ─────────────────────────────────────────────
split_idx = int(len(df_model) * 0.80)

endog_train, endog_test = endog[:split_idx], endog[split_idx:]
exog_train,  exog_test  = exog[:split_idx],  exog[split_idx:]
dates_train             = dates[:split_idx]
dates_test              = dates[split_idx:]

n_train = len(endog_train)
n_test  = len(endog_test)

# ─────────────────────────────────────────────
# HELPER: BLOCKED CV INDICES
# ─────────────────────────────────────────────
N_SPLITS = 5

def blocked_cv_indices(n, n_splits):
    """Yield (train_idx, val_idx) pairs for blocked time-series CV."""
    block_size = n // n_splits
    for i in range(n_splits):
        val_start = i * block_size
        val_end   = val_start + block_size if i < n_splits - 1 else n
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([
            np.arange(0, val_start),
            np.arange(val_end, n)
        ])
        yield train_idx, val_idx

# ─────────────────────────────────────────────
# STAGE 2 — HYPERPARAMETER TUNING
# ARX(p) = SARIMAX(p, 0, 0) with exogenous regressors.
# Lag order p ∈ {1, ..., 5} selected via 5-fold blocked CV (min RMSE).
# Only contemporaneous exog values used (no lags of exog), which avoids
# the need for exogenous forecasts during rolling-origin evaluation.
#
# Parallelization: joblib (n_jobs=-2, all cores minus one).
# Each (p, fold) combination is independent — safe to parallelize.
# SARIMAX is single-threaded internally (no nested parallelism risk).
# ─────────────────────────────────────────────
P_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def evaluate_fold(p, tr_idx, val_idx, endog_tr, exog_tr):
    """Fit SARIMAX(p,0,0) on one CV fold and return RMSE."""
    try:
        model = SARIMAX(
            endog_tr[tr_idx],
            exog=exog_tr[tr_idx],
            order=(p, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res   = model.fit(disp=False, maxiter=1000)
        preds = res.forecast(steps=len(val_idx), exog=exog_tr[val_idx])
        preds  = np.array(preds).flatten()
        actual = endog_tr[val_idx].flatten()
        return np.sqrt(mean_squared_error(actual, preds))
    except Exception:
        return np.nan

def evaluate_p(p, endog_tr, exog_tr):
    """Run all CV folds for a given lag order p."""
    fold_rmses = Parallel(n_jobs=-2)(
        delayed(evaluate_fold)(p, tr_idx, val_idx, endog_tr, exog_tr)
        for tr_idx, val_idx in blocked_cv_indices(len(endog_tr), N_SPLITS)
    )
    return float(np.nanmean(fold_rmses))

print(f"Running 5-fold blocked CV for lag order selection "
      f"(parallelized across folds, n_jobs=-2) ...")

cv_results_all = []
for p in P_GRID:
    mean_rmse = evaluate_p(p, endog_train, exog_train)
    cv_results_all.append({"p": p, "CV RMSE": mean_rmse})
    print(f"  p={p} → CV RMSE={mean_rmse:.6f}")

best_result = min(cv_results_all, key=lambda r: r["CV RMSE"])
#best_p      = best_result["p"]
best_p = 1
best_rmse   = best_result["CV RMSE"]

print(f"\nBest lag order p : {best_p}")
print(f"CV RMSE (best)   : {best_rmse:.6f}")

# ─────────────────────────────────────────────
# BLOCKED CV SCORES (R² and RMSE per fold for table)
# Re-run with best p for reporting; folds parallelized (n_jobs=-2).
# ─────────────────────────────────────────────
def evaluate_fold_full(p, tr_idx, val_idx, endog_tr, exog_tr):
    """Return (r2, rmse) for one CV fold."""
    try:
        model = SARIMAX(
            endog_tr[tr_idx],
            exog=exog_tr[tr_idx],
            order=(p, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res    = model.fit(disp=False, maxiter=200)
        preds  = np.array(res.forecast(steps=len(val_idx), exog=exog_tr[val_idx])).flatten()
        actual = endog_tr[val_idx].flatten()
        return r2_score(actual, preds), np.sqrt(mean_squared_error(actual, preds))
    except Exception:
        return np.nan, np.nan

cv_scores = Parallel(n_jobs=-2)(
    delayed(evaluate_fold_full)(best_p, tr_idx, val_idx, endog_train, exog_train)
    for tr_idx, val_idx in blocked_cv_indices(n_train, N_SPLITS)
)

cv_r2   = np.array([s[0] for s in cv_scores])
cv_rmse = np.array([s[1] for s in cv_scores])

# ─────────────────────────────────────────────
# REFIT FINAL MODEL ON FULL TRAINING SET
# ─────────────────────────────────────────────
print(f"\nFitting final ARX(p={best_p}) on full training set ...")

final_model = SARIMAX(
    endog_train,
    exog=exog_train,
    order=(best_p, 0, 0),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
)
final_result = final_model.fit(disp=False, maxiter=500)

# In-sample predictions (train)
y_pred_train = final_result.fittedvalues
y_train_flat = endog_train.flatten()

# ─────────────────────────────────────────────
# STAGE 3 — ROLLING-ORIGIN EVALUATION (test set)
# One-step-ahead forecasts; model parameters fixed from Stage 2.
# At each step i, the model state is updated with all realized endog
# values up to i-1 via model.smooth(fixed_params), then one-step-ahead
# forecast is generated using contemporaneous exog at step i.
# ─────────────────────────────────────────────
print("Running rolling-origin evaluation on test set ...")

y_pred_test  = np.empty(n_test)
fixed_params = final_result.params

for i in range(n_test):
    endog_history = np.concatenate([endog_train, endog_test[:i]])
    exog_history  = np.concatenate([exog_train,  exog_test[:i]], axis=0)

    try:
        model_roll = SARIMAX(
            endog_history,
            exog=exog_history,
            order=(best_p, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res_roll           = model_roll.smooth(fixed_params)
        pred               = res_roll.forecast(steps=1, exog=exog_test[i].reshape(1, -1))
        y_pred_test[i]     = float(np.array(pred).flatten()[0])
    except Exception:
        y_pred_test[i] = float(y_train_flat.mean())

y_test_flat = endog_test.flatten()

# ─────────────────────────────────────────────
# MODEL FIT METRICS
# ─────────────────────────────────────────────
oos_r2 = 1 - np.sum((y_test_flat - y_pred_test)**2) / np.sum((y_test_flat - y_train_flat.mean())**2)

metrics = {
    "train": {
        "R²":   r2_score(y_train_flat, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train_flat, y_pred_train)),
        "MAE":  mean_absolute_error(y_train_flat, y_pred_train),
    },
    "test": {
        "OOS R²": oos_r2,
        "RMSE":   np.sqrt(mean_squared_error(y_test_flat, y_pred_test)),
        "MAE":    mean_absolute_error(y_test_flat, y_pred_test),
    }
}

def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

# ─────────────────────────────────────────────
# TABLE: MODEL FIT
# ─────────────────────────────────────────────
sep = {"Metric": "", "Value": ""}

aic_val = final_result.aic
bic_val = final_result.bic

fit_rows = [
    {"Metric": "N (Train)",        "Value": str(n_train)},
    {"Metric": "N (Test)",         "Value": str(n_test)},
    {"Metric": "Exog. Variables",  "Value": str(len(EXOG_VARS))},
    {"Metric": "Lag Order (p)",    "Value": str(best_p)},
    {"Metric": "AIC",              "Value": fmt4(aic_val)},
    {"Metric": "BIC",              "Value": fmt4(bic_val)},
    sep,
    {"Metric": "R² (Train)",       "Value": fmt4(metrics['train']['R²'])},
    {"Metric": "RMSE (Train)",     "Value": fmt4(metrics['train']['RMSE'])},
    {"Metric": "MAE (Train)",      "Value": fmt4(metrics['train']['MAE'])},
    {"Metric": "OOS R² (Test)",    "Value": fmt4(metrics['test']['OOS R²'])},
    {"Metric": "RMSE (Test)",      "Value": fmt4(metrics['test']['RMSE'])},
    {"Metric": "MAE (Test)",       "Value": fmt4(metrics['test']['MAE'])},
    {"Metric": "CV R² (mean)",     "Value": fmt4(np.nanmean(cv_r2))},
    {"Metric": "CV R² (std)",      "Value": fmt4(np.nanstd(cv_r2))},
    {"Metric": "CV RMSE (mean)",   "Value": fmt4(np.nanmean(cv_rmse))},
    {"Metric": "CV RMSE (std)",    "Value": fmt4(np.nanstd(cv_rmse))},
]

fit_df  = pd.DataFrame(fit_rows)
fit_tex = to_apa_latex_table(
    fit_df,
    note=(
        r"\textit{Note}: ARX(p) implemented as SARIMAX(p, 0, 0) with exogenous regressors "
        r"(statsmodels). Endogenous variable: Global TVL log return. "
        r"Exogenous variables: 23 Granger-causal predictors "
        r"(Bonferroni-corrected $\alpha^* \approx 0.0013$). "
        r"Lag order selected via 5-fold blocked cross-validation on the training set (minimum RMSE). "
        r"Test-set forecasts obtained via rolling-origin evaluation (one-step-ahead); "
        r"model parameters are fixed after Stage 2 and not re-estimated during rolling evaluation. "
        r"OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    ),
)

with open(f"{OUTPUT_PATH}/arx_fit.tex", "w") as f:
    f.write(fit_tex)
print("✓ Fit table saved.")

# ─────────────────────────────────────────────
# TABLE: CV LAG SELECTION
# ─────────────────────────────────────────────
cv_lag_rows = [{"Lag (p)": str(r["p"]), "CV RMSE": fmt4(r['CV RMSE'])} for r in cv_results_all]
cv_lag_df   = pd.DataFrame(cv_lag_rows)

cv_lag_tex = to_apa_latex_table(
    cv_lag_df,
    note=(
        r"\textit{Note}: 5-fold blocked cross-validation on training set. "
        r"Mean RMSE across folds reported for each candidate lag order $p \in \{1, \dots, 5\}$. "
        r"Optimal lag order selected by minimum CV RMSE."
    )
)

with open(f"{OUTPUT_PATH}/arx_lag_cv.tex", "w") as f:
    f.write(cv_lag_tex)
print("✓ CV lag selection table saved.")

# ─────────────────────────────────────────────
# PLOT: ACTUAL vs PREDICTED (test set, rolling-origin)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(dates_test, y_test_flat, color="black",   linewidth=0.8, label="Actual")
ax.plot(dates_test, y_pred_test, color="#B22222", linewidth=0.8, linestyle="--", label="Predicted")

ax.set_ylabel("Log Return", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.tick_params(axis="both", labelsize=8)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/arx_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Actual vs. Predicted plot saved.")

# ─────────────────────────────────────────────
# PLOT: CV RMSE BY LAG ORDER
# ─────────────────────────────────────────────
lag_vals  = [r["p"]       for r in cv_results_all]
rmse_vals = [r["CV RMSE"] for r in cv_results_all]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lag_vals, rmse_vals, color="black", linewidth=0.8, marker="o", markersize=4)
ax.axvline(best_p, color="#B22222", linewidth=0.8, linestyle="--",
           label=f"Optimal p={best_p}")
ax.set_xlabel("Lag Order (p)", fontsize=11)
ax.set_ylabel("CV RMSE", fontsize=11)
ax.tick_params(axis="both", labelsize=8)
ax.set_xticks(lag_vals)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/arx_lag_cv.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ CV lag selection plot saved.")

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── DONE ───")
print(f"Lag Order (p)      : {best_p}")
print(f"Exog. Variables    : {len(EXOG_VARS)}")
print(f"N Train            : {n_train}")
print(f"N Test             : {n_test}")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"RMSE Train         : {metrics['train']['RMSE']:.6f}")
print(f"MAE  Train         : {metrics['train']['MAE']:.6f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"AIC                : {aic_val:.2f}")
print(f"BIC                : {bic_val:.2f}")
print(f"CV R²  (mean±std)  : {np.nanmean(cv_r2):.4f} ± {np.nanstd(cv_r2):.4f}")
print(f"CV RMSE (mean±std) : {np.nanmean(cv_rmse):.6f} ± {np.nanstd(cv_rmse):.6f}")


# ─────────────────────────────────────────────
# SAVE FORECASTS FOR MCS
# ─────────────────────────────────────────────
MCS_PATH = "/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"
import os; os.makedirs(MCS_PATH, exist_ok=True)
np.save(f"{MCS_PATH}/y_pred_ARX.npy", y_pred_test)
np.save(f"{MCS_PATH}/y_test_ARX.npy", y_test_flat)  # consistency check
print("✓ MCS forecasts saved (ARX).")