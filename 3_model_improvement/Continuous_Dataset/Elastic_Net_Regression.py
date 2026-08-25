import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
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
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/3_model_improvement/Tables_Figures"

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

# ─────────────────────────────────────────────
# SCALING
# Fit scaler exclusively on training data to prevent data leakage.
# ─────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STAGE 2 — HYPERPARAMETER TUNING
# 5-fold blocked (non-shuffled) cross-validation on training set.
# Each fold is a contiguous, non-overlapping block preserving temporal order.
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

# Manual blocked CV to select best (alpha, l1_ratio) by mean RMSE
alphas    = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

best_rmse, best_alpha, best_l1 = np.inf, None, None

for alpha in alphas:
    for l1 in l1_ratios:
        fold_rmses = []
        for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
            # Scale within each fold to avoid leakage across blocks
            sc_fold   = StandardScaler()
            Xtr_fold  = sc_fold.fit_transform(X_train[tr_idx])
            Xval_fold = sc_fold.transform(X_train[val_idx])

            model = ElasticNet(alpha=alpha, l1_ratio=l1,
                               max_iter=10000, random_state=42)
            model.fit(Xtr_fold, y_train[tr_idx])
            preds = model.predict(Xval_fold)
            fold_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))

        mean_rmse = np.mean(fold_rmses)
        if mean_rmse < best_rmse:
            best_rmse  = mean_rmse
            best_alpha = alpha
            best_l1    = l1

print(f"Best alpha    : {best_alpha}")
print(f"Best l1_ratio : {best_l1}")
print(f"CV RMSE (best): {best_rmse:.6f}")

# ─────────────────────────────────────────────
# REFIT FINAL MODEL ON FULL TRAINING SET
# ─────────────────────────────────────────────
enet = ElasticNet(alpha=best_alpha, l1_ratio=best_l1,
                  max_iter=10000, random_state=42)
enet.fit(X_train_sc, y_train)

# In-sample predictions (train)
y_pred_train = enet.predict(X_train_sc)

# ─────────────────────────────────────────────
# STAGE 3 — ROLLING-ORIGIN EVALUATION (test set)
# Starting at the first test observation, the origin advances by one
# period each step, generating a series of one-step-ahead predictions.
# The model is NOT re-fitted; coefficients are fixed from Stage 2.
# ─────────────────────────────────────────────
y_pred_rolling = np.empty(len(y_test))
for i in range(len(y_test)):
    y_pred_rolling[i] = enet.predict(X_test_sc[i].reshape(1, -1))[0]

y_pred_test = y_pred_rolling   # alias for convenience

# ─────────────────────────────────────────────
# HC3 STANDARD ERRORS  (via OLS on Elastic Net selected features)
# ─────────────────────────────────────────────
nonzero_mask      = enet.coef_ != 0
selected_features = [FEATURES[i] for i, m in enumerate(nonzero_mask) if m]

X_train_sel = X_train_sc[:, nonzero_mask]
X_train_sm  = sm.add_constant(X_train_sel)

ols_model  = sm.OLS(y_train, X_train_sm).fit(cov_type="HC3")
coef_names = ["Intercept"] + selected_features
coefs      = ols_model.params
std_errors = ols_model.HC3_se
t_stats    = ols_model.tvalues
p_values   = ols_model.pvalues

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

# ─────────────────────────────────────────────
# BLOCKED CV SCORES (for table reporting)
# Re-run with fixed best hyperparameters for R² and RMSE per fold.
# ─────────────────────────────────────────────
cv_r2_scores   = []
cv_rmse_scores = []

for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
    sc_fold   = StandardScaler()
    Xtr_fold  = sc_fold.fit_transform(X_train[tr_idx])
    Xval_fold = sc_fold.transform(X_train[val_idx])

    m = ElasticNet(alpha=best_alpha, l1_ratio=best_l1,
                   max_iter=10000, random_state=42)
    m.fit(Xtr_fold, y_train[tr_idx])
    preds = m.predict(Xval_fold)

    cv_r2_scores.append(r2_score(y_train[val_idx], preds))
    cv_rmse_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))

cv_r2   = np.array(cv_r2_scores)
cv_rmse = np.array(cv_rmse_scores)

# ─────────────────────────────────────────────
# MODEL FIT METRICS
# OOS R² = 1 - SS_res / SS_tot  (sklearn convention, applied to test set)
# ─────────────────────────────────────────────
def adj_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

n_train_n = len(y_train)
n_test_n  = len(y_test)
k         = int(nonzero_mask.sum())

oos_r2 = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_train))**2)   # 1 - SS_res / SS_tot

metrics = {
    "train": {
        "R²":      r2_score(y_train, y_pred_train),
        "Adj. R²": adj_r2(r2_score(y_train, y_pred_train), n_train_n, k),
        "RMSE":    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "MAE":     mean_absolute_error(y_train, y_pred_train),
    },
    "test": {
        "OOS R²":  oos_r2,
        "RMSE":    np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE":     mean_absolute_error(y_test, y_pred_test),
    }
}

# ─────────────────────────────────────────────
# TABLE 1: COEFFICIENTS + MODEL FIT
# ─────────────────────────────────────────────
coef_rows = []
for name, coef, se, t, p in zip(coef_names, coefs, std_errors, t_stats, p_values):
    stars = sig_stars(p)
    coef_rows.append({
        "Variable":        name,
        "$\\hat{\\beta}$": fmt4(coef) + stars,
        "Std. Error":      fmt4(se),
        "t-Statistic":     fmt4(t),
        "p-Value":         format_pval(p)
    })

sep = {"Variable": "", "$\\hat{\\beta}$": "", "Std. Error": "", "t-Statistic": "", "p-Value": ""}

fit_rows = [
    sep,
    {"Variable": "N (Train)",         "$\\hat{\\beta}$": str(n_train_n),                                   "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "N (Test)",          "$\\hat{\\beta}$": str(n_test_n),                                    "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "Selected Vars",     "$\\hat{\\beta}$": str(k),                                           "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "$\\alpha$",         "$\\hat{\\beta}$": f"{best_alpha}",                                  "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "l1_ratio",        "$\\hat{\\beta}$": f"{best_l1}",                                     "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "R² (Train)",        "$\\hat{\\beta}$": fmt4(metrics['train']['R²']),                     "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "Adj. R² (Train)",   "$\\hat{\\beta}$": fmt4(metrics['train']['Adj. R²']),                "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "RMSE (Train)",      "$\\hat{\\beta}$": fmt4(metrics['train']['RMSE']),                   "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "MAE (Train)",       "$\\hat{\\beta}$": fmt4(metrics['train']['MAE']),                    "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "OOS R² (Test)",     "$\\hat{\\beta}$": fmt4(metrics['test']['OOS R²']),                 "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "RMSE (Test)",       "$\\hat{\\beta}$": fmt4(metrics['test']['RMSE']),                    "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "MAE (Test)",        "$\\hat{\\beta}$": fmt4(metrics['test']['MAE']),                     "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV R² (mean)",      "$\\hat{\\beta}$": fmt4(cv_r2.mean()),                               "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV R² (std)",       "$\\hat{\\beta}$": fmt4(cv_r2.std()),                                "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV RMSE (mean)",    "$\\hat{\\beta}$": fmt4(cv_rmse.mean()),                             "Std. Error": "", "t-Statistic": "", "p-Value": ""},
    {"Variable": "CV RMSE (std)",     "$\\hat{\\beta}$": fmt4(cv_rmse.std()),                              "Std. Error": "", "t-Statistic": "", "p-Value": ""},
]

coef_df = pd.DataFrame(coef_rows + fit_rows)

coef_tex = to_apa_latex_table(
    coef_df,
    note=(
        r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1. "
        r"HC3 heteroskedasticity-robust standard errors. "
        r"Coefficients estimated on standardized features. "
        r"Train/Test split: 80/20 (chronological). "
        r"Hyperparameters selected via 5-fold blocked cross-validation on the training set. "
        r"Test-set forecasts obtained via rolling-origin evaluation (one-step-ahead). "
        r"OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    ),
    col_rename={"$\\hat{\\beta}$": "$\\hat{\\beta}$"}
)

with open(f"{OUTPUT_PATH}/enet_coef_HC3.tex", "w") as f:
    f.write(coef_tex)
print("✓ Coefficient table saved.")

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
plt.savefig(f"{OUTPUT_PATH}/enet_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Actual vs. Predicted plot saved.")

# ─────────────────────────────────────────────
# PLOT: COEFFICIENT PATH (non-zero only)
# ─────────────────────────────────────────────
coef_plot_df = pd.DataFrame({
    "Variable":    selected_features,
    "Coefficient": enet.coef_[nonzero_mask]
}).sort_values("Coefficient")

fig, ax = plt.subplots(figsize=(10, max(4, len(selected_features) * 0.35)))
colors = ["#B22222" if c < 0 else "#2C3E50" for c in coef_plot_df["Coefficient"]]
ax.barh(coef_plot_df["Variable"], coef_plot_df["Coefficient"], color=colors, edgecolor="none")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardized Coefficient", fontsize=11)
ax.tick_params(axis="both", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/enet_coefficients.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Coefficient plot saved.")

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── DONE ───")
print(f"Selected features  : {k} / {len(FEATURES)}")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"CV R²  (mean±std)  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"CV RMSE (mean±std) : {cv_rmse.mean():.6f} ± {cv_rmse.std():.6f}")

# ─────────────────────────────────────────────
# TABLE 2: SELECTED VARIABLES (compact, sign only)
# ─────────────────────────────────────────────
sel_rows = []
for feat, coef in sorted(
    zip(selected_features, enet.coef_[nonzero_mask]),
    key=lambda x: -x[1]
):
    sign = "$+$" if coef > 0 else "$-$"
    sel_rows.append({"Variable": feat, "Sign": sign})

sel_df  = pd.DataFrame(sel_rows)
sel_tex = to_apa_latex_table(
    sel_df,
    note=(
        r"\textit{Note}: Variables selected by Elastic Net regularization "
        r"($\alpha=0.001$, l1\_ratio$=0.7$) via 5-fold blocked cross-validation. "
        r"Sign denotes the direction of the standardized coefficient. "
        r"14 out of 38 candidate features were retained."
    )
)
with open(f"{OUTPUT_PATH}/enet_selected_vars.tex", "w") as f:
    f.write(sel_tex)
print("✓ Selected variables table saved.")

# ─────────────────────────────────────────────
# SAVE FORECASTS FOR MCS
# ─────────────────────────────────────────────
MCS_PATH = "/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"
import os; os.makedirs(MCS_PATH, exist_ok=True)
np.save(f"{MCS_PATH}/y_pred_ENet.npy", y_pred_test)
print("✓ MCS forecasts saved (ENet).")