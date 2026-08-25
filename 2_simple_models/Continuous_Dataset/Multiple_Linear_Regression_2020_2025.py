import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/2_simple_models/Tables_Figures"

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

TARGET   = "Global_TVL_USD_log_return"
FEATURES = [c for c in df.columns if c not in ["date", TARGET]]

df_model = df[["date", TARGET] + FEATURES].dropna()

X     = df_model[FEATURES].values
y     = df_model[TARGET].values
dates = df_model["date"].values

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT  (80 / 20, chronological)
# ─────────────────────────────────────────────
split_idx = int(len(df_model) * 0.80)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train     = dates[:split_idx]
dates_test      = dates[split_idx:]

n_train, k = X_train.shape

# ─────────────────────────────────────────────
# STAGE 2 — NO HYPERPARAMETER TUNING (OLS)
# MLR has no regularization hyperparameters.
# Blocked CV is therefore conducted for in-sample
# performance reporting only (R², RMSE per fold).
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

n_train_obs = len(y_train)

cv_r2_scores   = []
cv_rmse_scores = []

for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
    X_tr_fold  = sm.add_constant(X_train[tr_idx],  has_constant="add")
    X_val_fold = sm.add_constant(X_train[val_idx], has_constant="add")

    m     = sm.OLS(y_train[tr_idx], X_tr_fold).fit()
    preds = m.predict(X_val_fold)

    cv_r2_scores.append(r2_score(y_train[val_idx], preds))
    cv_rmse_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))

cv_r2   = np.array(cv_r2_scores)
cv_rmse = np.array(cv_rmse_scores)

# ─────────────────────────────────────────────
# STAGE 2 — FIT HC3 OLS ON FULL TRAINING SET
# ─────────────────────────────────────────────
X_sm_train = sm.add_constant(X_train, has_constant="add")
ols_result = sm.OLS(y_train, X_sm_train).fit(cov_type="HC3")

beta       = ols_result.params
se         = ols_result.bse
t_stats    = ols_result.tvalues
p_values   = ols_result.pvalues
r2_train   = ols_result.rsquared
adj_r2_val = ols_result.rsquared_adj
f_stat     = ols_result.fvalue
f_pval     = ols_result.f_pvalue
aic        = ols_result.aic
bic        = ols_result.bic

residuals_train = ols_result.resid
y_pred_train    = ols_result.fittedvalues

# ─────────────────────────────────────────────
# STAGE 3 — ROLLING-ORIGIN EVALUATION (test set)
# For a purely exogenous MLR (no lagged DV), one-step-ahead
# rolling-origin predictions reduce to applying the fixed
# training model sequentially to each test observation.
# ─────────────────────────────────────────────
X_sm_test = sm.add_constant(X_test, has_constant="add")

y_pred_rolling = np.empty(len(y_test))
for i in range(len(y_test)):
    y_pred_rolling[i] = ols_result.predict(X_sm_test[i].reshape(1, -1))[0]

y_pred_test = y_pred_rolling

# ─────────────────────────────────────────────
# MODEL FIT METRICS
# ─────────────────────────────────────────────
n_train_n = len(y_train)
n_test_n  = len(y_test)

metrics = {
    "train": {
        "R²":      r2_train,
        "Adj. R²": adj_r2_val,
        "RMSE":    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "MAE":     mean_absolute_error(y_train, y_pred_train),
    },
    "test": {
        "OOS R²": 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_train))**2),
        "RMSE":   np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE":    mean_absolute_error(y_test, y_pred_test),
    }
}

# ─────────────────────────────────────────────
# HELPER: SIGNIFICANCE STARS
# ─────────────────────────────────────────────
def sig_stars(p):
    if p < 0.01:  return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""

def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

def format_pval(p):
    return r"$<$0.0001" if p < 0.0001 else f"{p:.4f}"

def format_coef(b, s):
    return fmt4(b) + s

# ─────────────────────────────────────────────
# TABLE 1: COEFFICIENTS + MODEL FIT
# ─────────────────────────────────────────────
coef_names = ["Intercept"] + FEATURES

coef_rows = []
for name, coef, s, t, p in zip(coef_names, beta, se, t_stats, p_values):
    stars = sig_stars(p)
    coef_rows.append({
        "Variable":        name,
        "$\\hat{\\beta}$": format_coef(coef, stars),
        "Std. Error":      fmt4(s),
        "t-Statistic":     fmt4(t),
        "p-Value":         format_pval(p)
    })

sep = {"Variable": "", "$\\hat{\\beta}$": "", "Std. Error": "", "t-Statistic": "", "p-Value": ""}

f_stars = sig_stars(f_pval)

fit_rows = [
    sep,
    {"Variable": "N (Train)",       "$\\hat{\\beta}$": str(n_train_n),                          "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "N (Test)",        "$\\hat{\\beta}$": str(n_test_n),                           "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "R² (Train)",      "$\\hat{\\beta}$": fmt4(metrics['train']['R²']),            "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "Adj. R² (Train)", "$\\hat{\\beta}$": fmt4(metrics['train']['Adj. R²']),       "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "RMSE (Train)",    "$\\hat{\\beta}$": fmt4(metrics['train']['RMSE']),          "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "MAE (Train)",     "$\\hat{\\beta}$": fmt4(metrics['train']['MAE']),           "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "F-Statistic",     "$\\hat{\\beta}$": fmt4(f_stat) + f_stars,                  "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "AIC",             "$\\hat{\\beta}$": fmt4(aic),                               "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "BIC",             "$\\hat{\\beta}$": fmt4(bic),                               "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "OOS R² (Test)",   "$\\hat{\\beta}$": fmt4(metrics['test']['OOS R²']),        "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "RMSE (Test)",     "$\\hat{\\beta}$": fmt4(metrics['test']['RMSE']),           "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "MAE (Test)",      "$\\hat{\\beta}$": fmt4(metrics['test']['MAE']),            "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV R² (mean)",    "$\\hat{\\beta}$": fmt4(cv_r2.mean()),                      "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV R² (std)",     "$\\hat{\\beta}$": fmt4(cv_r2.std()),                       "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV RMSE (mean)",  "$\\hat{\\beta}$": fmt4(cv_rmse.mean()),                    "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV RMSE (std)",   "$\\hat{\\beta}$": fmt4(cv_rmse.std()),                     "Std. Error": "", "t-Statistic": "", "p-Value": ""},
]

coef_df = pd.DataFrame(coef_rows + fit_rows)

coef_tex = to_apa_latex_table(
    coef_df,
    note=(
        r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1. "
        r"HC3 heteroskedasticity-robust standard errors. "
        r"Estimated on training set (first 80\% of observations). "
        r"Test-set forecasts obtained via rolling-origin evaluation (one-step-ahead). "
        r"OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    )
)

# Table is long (39 coefficient + 16 model-fit rows); tighten row height so
# it fits within \textheight on a single page (same fix as granger_causality.tex).
coef_tex = coef_tex.replace(
    r"\begin{tabular}", "{\\renewcommand{\\arraystretch}{0.90}\n\\begin{tabular}", 1
).replace(
    r"\end{tabular}", r"\end{tabular}}", 1
)

with open(f"{OUTPUT_PATH}/mlrcoef_HC3_20202025.tex", "w") as f:
    f.write(coef_tex)
print("✓ Coefficient table saved.")

# ─────────────────────────────────────────────
# WHITE TEST FOR HETEROSKEDASTICITY (train residuals)
# ─────────────────────────────────────────────
white_stat, white_pval, white_f, white_fpval = het_white(residuals_train, X_sm_train)

white_df = pd.DataFrame({
    "Test":         ["White Test"],
    "LM Statistic": [fmt4(white_stat)],
    "LM p-Value":   [format_pval(white_pval)],
    "F-Statistic":  [fmt4(white_f)],
    "F p-Value":    [format_pval(white_fpval)]
})

white_tex = to_apa_latex_table(
    white_df,
    note=(
        r"\textit{Note}: H\textsubscript{0}: Homoskedasticity (constant variance of residuals). "
        r"H\textsubscript{1}: Heteroskedasticity. "
        r"Significance level: $\alpha = 0.05$. Conducted on training set residuals."
    )
)

with open(f"{OUTPUT_PATH}/white_test_20202025.tex", "w") as f:
    f.write(white_tex)
print("✓ White test table saved.")

# ─────────────────────────────────────────────
# PLOT: ACTUAL vs PREDICTED (test set, rolling-origin)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(dates_test, y_test,      color="black",   linewidth=0.8, label="Actual")
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
plt.savefig(f"{OUTPUT_PATH}/mlr_actual_vs_predicted_test_20202025.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Actual vs. Predicted plot saved.")

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── DONE ───")
print(f"N Train            : {n_train_n}")
print(f"N Test             : {n_test_n}")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"Adj. R² Train      : {metrics['train']['Adj. R²']:.4f}")
print(f"RMSE Train         : {metrics['train']['RMSE']:.6f}")
print(f"MAE  Train         : {metrics['train']['MAE']:.6f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"CV R²  (mean±std)  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"CV RMSE (mean±std) : {cv_rmse.mean():.6f} ± {cv_rmse.std():.6f}")
print(f"White Test — LM: {white_stat:.4f} (p={'< 0.0001' if white_pval < 0.0001 else f'{white_pval:.4f}'})  |  "
      f"F: {white_f:.4f} (p={'< 0.0001' if white_fpval < 0.0001 else f'{white_fpval:.4f}'})")

# ─────────────────────────────────────────────
# SAVE FORECASTS FOR MCS
# ─────────────────────────────────────────────
MCS_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"
import os; os.makedirs(MCS_PATH, exist_ok=True)
np.save(f"{MCS_PATH}/y_pred_MLR.npy",  y_pred_test)
np.save(f"{MCS_PATH}/y_test.npy",      y_test)
print("✓ MCS forecasts saved (MLR).")