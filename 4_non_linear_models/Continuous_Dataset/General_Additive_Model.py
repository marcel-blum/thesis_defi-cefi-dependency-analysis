import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pygam import LinearGAM, s, l
from sklearn.preprocessing import StandardScaler
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
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/4_non_linear_models/Tables_Figures"

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

TARGET = "Global_TVL_USD_log_return"

# All candidate features (38 variables)
SELECTED_VARS = [c for c in df.columns if c not in ["date", TARGET,
    "ETH_x_XRP", "XRP_x_XRPVol", "Stablecoins_x_Revenue", "Liquidity_x_CPI"]]

# Interaction terms (product features, linear approximation of tensor product splines).
# All pairs share the same coefficient sign (directional consistency).
df["ETH_x_XRP"]            = df["ETHUSDT_Log_Return"]   * df["XRPUSDT_Log_Return"]      # (+) x (+)
df["XRP_x_XRPVol"]         = df["XRPUSDT_Log_Return"]   * df["XRPUSDT_Vol_LogGrowth"]   # (+) x (+)
df["Stablecoins_x_Revenue"] = df["Global_Stablecoins_Mcap_USD_log_return"] * df["Global_Gross_Revenue_USD_log_return"]  # (+) x (+)
df["Liquidity_x_CPI"]      = df["USA_D_Liquidity"]      * df["USA_D_CPI"]               # (-) x (-)

INTERACTION_VARS = ["ETH_x_XRP", "XRP_x_XRPVol", "Stablecoins_x_Revenue", "Liquidity_x_CPI"]
ALL_FEATURES     = SELECTED_VARS + INTERACTION_VARS

df_model = df[["date", TARGET] + ALL_FEATURES].dropna()

X     = df_model[ALL_FEATURES].values
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
# HELPER: BUILD GAM TERMS
# ─────────────────────────────────────────────
n_main = len(SELECTED_VARS)
n_int  = len(INTERACTION_VARS)

def build_terms(n_sp, lam_val):
    term_list = [s(i, n_splines=n_sp, lam=lam_val) for i in range(n_main)] \
              + [l(i) for i in range(n_main, n_main + n_int)]
    terms = term_list[0]
    for t in term_list[1:]:
        terms += t
    return terms

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

n_train_obs = len(y_train)

# ─────────────────────────────────────────────
# STAGE 2 — HYPERPARAMETER TUNING
# 5-fold blocked CV on training set.
# ─────────────────────────────────────────────
n_splines_grid = [4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
lam_grid       = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

best_rmse, best_n_sp, best_lam = np.inf, None, None

print("Running 5-fold blocked CV for hyperparameter tuning ...")
for n_sp in n_splines_grid:
    for lam_val in lam_grid:
        fold_rmses = []
        for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
            sc_fold   = StandardScaler()
            Xtr_fold  = sc_fold.fit_transform(X_train[tr_idx])
            Xval_fold = sc_fold.transform(X_train[val_idx])

            gam = LinearGAM(build_terms(n_sp, lam_val))
            gam.fit(Xtr_fold, y_train[tr_idx])
            preds = gam.predict(Xval_fold)
            fold_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))

        mean_rmse = np.mean(fold_rmses)
        print(f"  n_splines={n_sp}, lam={lam_val} → CV RMSE={mean_rmse:.6f}")
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_n_sp = n_sp
            best_lam  = lam_val

print(f"\nBest n_splines : {best_n_sp}")
print(f"Best lam       : {best_lam}")
print(f"CV RMSE (best) : {best_rmse:.6f}")

# ─────────────────────────────────────────────
# REFIT FINAL MODEL ON FULL TRAINING SET
# ─────────────────────────────────────────────
gam_final = LinearGAM(build_terms(best_n_sp, best_lam))
gam_final.fit(X_train_sc, y_train)

# In-sample predictions (train)
y_pred_train = gam_final.predict(X_train_sc)

# ─────────────────────────────────────────────
# STAGE 3 — ROLLING-ORIGIN EVALUATION (test set)
# Starting at the first test observation, the origin advances by one
# period each step, generating a series of one-step-ahead predictions.
# The model is NOT re-fitted; coefficients are fixed from Stage 2.
# ─────────────────────────────────────────────
y_pred_rolling = np.empty(len(y_test))
for i in range(len(y_test)):
    y_pred_rolling[i] = gam_final.predict(X_test_sc[i].reshape(1, -1))[0]

y_pred_test = y_pred_rolling

# ─────────────────────────────────────────────
# BLOCKED CV SCORES (for table reporting)
# Re-run with best hyperparameters for R² and RMSE per fold.
# ─────────────────────────────────────────────
cv_r2_scores   = []
cv_rmse_scores = []

for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
    sc_fold   = StandardScaler()
    Xtr_fold  = sc_fold.fit_transform(X_train[tr_idx])
    Xval_fold = sc_fold.transform(X_train[val_idx])

    gam_cv = LinearGAM(build_terms(best_n_sp, best_lam))
    gam_cv.fit(Xtr_fold, y_train[tr_idx])
    preds = gam_cv.predict(Xval_fold)

    cv_r2_scores.append(r2_score(y_train[val_idx], preds))
    cv_rmse_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))

cv_r2   = np.array(cv_r2_scores)
cv_rmse = np.array(cv_rmse_scores)

# ─────────────────────────────────────────────
# MODEL FIT METRICS
# ─────────────────────────────────────────────
n_train_n = len(y_train)
n_test_n  = len(y_test)
oos_r2 = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_train))**2)

metrics = {
    "train": {
        "R²":   r2_score(y_train, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "MAE":  mean_absolute_error(y_train, y_pred_train),
    },
    "test": {
        "OOS R²": oos_r2,
        "RMSE":   np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE":    mean_absolute_error(y_test, y_pred_test),
    }
}

# ─────────────────────────────────────────────
# ADJUSTED R² (manual, no native GAM support)
# ─────────────────────────────────────────────
k_terms      = n_main + n_int
adj_r2_train = 1 - (1 - metrics["train"]["R²"]) * (n_train_n - 1) / (n_train_n - k_terms - 1)

# ─────────────────────────────────────────────
# HELPER: SIGNIFICANCE STARS
# ─────────────────────────────────────────────
def sig_stars(p):
    if p < 0.01:   return "***"
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

# ─────────────────────────────────────────────
# TABLE: MODEL FIT
# ─────────────────────────────────────────────
sep = {"Metric": "", "Value": ""}

fit_rows = [
    {"Metric": "N (Train)",        "Value": str(n_train_n)},
    {"Metric": "N (Test)",         "Value": str(n_test_n)},
    {"Metric": "Selected Vars",    "Value": str(len(SELECTED_VARS))},
    {"Metric": "Interaction Terms","Value": str(len(INTERACTION_VARS))},
    {"Metric": "n\\_splines",      "Value": str(best_n_sp)},
    {"Metric": "lam",              "Value": str(best_lam)},
    sep,
    {"Metric": "R² (Train)",       "Value": fmt4(metrics['train']['R²'])},
    {"Metric": "Adj. R² (Train)",  "Value": fmt4(adj_r2_train)},
    {"Metric": "RMSE (Train)",     "Value": fmt4(metrics['train']['RMSE'])},
    {"Metric": "MAE (Train)",      "Value": fmt4(metrics['train']['MAE'])},
    {"Metric": "OOS R² (Test)",    "Value": fmt4(metrics['test']['OOS R²'])},
    {"Metric": "RMSE (Test)",      "Value": fmt4(metrics['test']['RMSE'])},
    {"Metric": "MAE (Test)",       "Value": fmt4(metrics['test']['MAE'])},
    {"Metric": "CV R² (mean)",     "Value": fmt4(cv_r2.mean())},
    {"Metric": "CV R² (std)",      "Value": fmt4(cv_r2.std())},
    {"Metric": "CV RMSE (mean)",   "Value": fmt4(cv_rmse.mean())},
    {"Metric": "CV RMSE (std)",    "Value": fmt4(cv_rmse.std())},
]

fit_df  = pd.DataFrame(fit_rows)
fit_tex = to_apa_latex_table(
    fit_df,
    note=(
        r"\textit{Note}: LinearGAM (pygam) with P-splines for main effects and "
        r"linear product terms for interactions. "
        r"Hyperparameters selected via 5-fold blocked cross-validation on the training set. "
        r"Test-set forecasts obtained via rolling-origin evaluation (one-step-ahead); "
        r"model coefficients are fixed after Stage 2 and not re-estimated during rolling evaluation. "
        r"OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    ),
)

with open(f"{OUTPUT_PATH}/gam_fit.tex", "w") as f:
    f.write(fit_tex)
print("✓ Fit table saved.")

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
plt.savefig(f"{OUTPUT_PATH}/gam_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Actual vs. Predicted plot saved.")

# ─────────────────────────────────────────────
# PERMUTATION IMPORTANCE
# GAM (pygam) provides no native feature importances. Permutation
# importance is the model-agnostic alternative: mean decrease in R²
# over n_repeats=10 random shuffles of each feature, computed on the
# held-out test set. Consistent with the approach used for SVR (RBF).
#
# pygam's LinearGAM does not inherit from sklearn.BaseEstimator and
# therefore cannot be passed directly to sklearn's permutation_importance.
# A lightweight sklearn-compatible wrapper is used instead.
# ─────────────────────────────────────────────
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance

class GAMWrapper(BaseEstimator, RegressorMixin):
    """Thin sklearn-compatible wrapper around a fitted pygam LinearGAM."""
    def __init__(self, gam):
        self.gam = gam
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.gam.predict(X)

gam_wrapped = GAMWrapper(gam_final)

perm_result = permutation_importance(
    gam_wrapped, X_test_sc, y_test,
    n_repeats=10, random_state=42, n_jobs=-2,
    scoring="r2"
)

importance_df = pd.DataFrame({
    "Variable":   ALL_FEATURES,
    "Perm. Mean": perm_result.importances_mean,
    "Perm. Std":  perm_result.importances_std,
}).sort_values("Perm. Mean", ascending=False).reset_index(drop=True)

imp_rows = []
for _, row in importance_df.iterrows():
    imp_rows.append({
        "Variable":         row["Variable"],
        "Perm. Importance": fmt4(row['Perm. Mean']),
        "Perm. Std.":       fmt4(row['Perm. Std']),
    })

imp_df_tex = pd.DataFrame(imp_rows)
imp_tex = to_apa_latex_table(
    imp_df_tex,
    note=(
        r"\textit{Note}: Permutation importance = mean decrease in $R^2$ over 10 repeats "
        r"when a feature is randomly shuffled, computed on the held-out test set. "
        r"GAM (pygam) does not provide native feature importances; "
        r"permutation importance is therefore the sole attribution metric reported. "
        r"Sorted descending by permutation importance."
    )
)

with open(f"{OUTPUT_PATH}/gam_importance.tex", "w") as f:
    f.write(imp_tex)
print("✓ Feature importance table saved.")

# ─────────────────────────────────────────────
# PLOT: PERMUTATION IMPORTANCE (top 20)
# ─────────────────────────────────────────────
top_n   = 20
plot_df = importance_df.head(top_n).sort_values("Perm. Mean")

fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
ax.barh(plot_df["Variable"], plot_df["Perm. Mean"],
        xerr=plot_df["Perm. Std"], color="#2C3E50", edgecolor="none",
        error_kw={"elinewidth": 0.8, "ecolor": "#888888", "capsize": 2})
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Permutation Importance (Mean Decrease in $R^2$)", fontsize=11)
ax.tick_params(axis="both", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/gam_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Permutation importance plot saved.")

# ─────────────────────────────────────────────
# PLOT: PARTIAL DEPENDENCE (grouped by variable category)
# ─────────────────────────────────────────────
PDP_GROUPS = {
    "01_On-Chain_Variables": [
        "Global_Stablecoins_Mcap_USD_log_return",
        "Global_Merchandise_Vol_USD_log_return",
        "Global_Gross_Revenue_USD_log_return",
    ],
    "02_Cryptocurrencies": [
        "BTCUSDT_Log_Return", "ETHUSDT_Log_Return", "BNBUSDT_Log_Return",
        "XRPUSDT_Log_Return", "SOLUSDT_Log_Return",
        "BTCUSDT_Vol_LogGrowth", "ETHUSDT_Vol_LogGrowth", "BNBUSDT_Vol_LogGrowth",
        "XRPUSDT_Vol_LogGrowth", "SOLUSDT_Vol_LogGrowth",
    ],
    "03_Exchange_Rates": [
        "EUR_USD_log_return", "EUR_CNY_log_return", "USD_CNY_log_return",
    ],
    "04_Interest_Rates": [
        "USA_Rate_Hike", "USA_Rate_Cut",
        "EZ_Rate_Hike", "EZ_Rate_Cut",
        "China_Rate_Cut",
    ],
    "05_Macros": [
        "USA_D_GDP", "EZ_D_GDP",
        "USA_D_CPI", "EZ_D_CPI", "CHN_D_CPI",
        "USA_D_UR",  "EZ_D_UR",  "CHN_D_UR",
    ],
    "06_Money_Supply": [
        "USA_D_Liquidity", "EZ_D_Liquidity", "CHN_D_Liquidity",
    ],
    "07_Indices": [
        "SP500_log_return", "STOXX600_log_return", "CSI300_log_return",
        "MSCIEM_log_return", "GlobalGold_log_return", "SOX_log_return",
    ],
}

def pdp_plot_group(group_name, var_list, selected_vars, gam_model, output_path):
    # Only plot vars present in SELECTED_VARS
    vars_to_plot = [v for v in var_list if v in selected_vars]
    if not vars_to_plot:
        return

    n_vars = len(vars_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, var in enumerate(vars_to_plot):
        term_idx = selected_vars.index(var)
        ax = axes[idx]
        XX = gam_model.generate_X_grid(term=term_idx)
        pdep, confi = gam_model.partial_dependence(term=term_idx, X=XX, width=0.95)
        ax.plot(XX[:, term_idx], pdep, color="black", linewidth=1.0)
        ax.fill_between(XX[:, term_idx], confi[:, 0], confi[:, 1], alpha=0.2, color="grey")
        ax.set_title(var.replace("_", " "), fontstyle="italic", fontsize=15, pad=6)
        ax.set_xlabel("Feature Value (Standardized)", fontsize=13)
        ax.set_ylabel("Partial Effect", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fname = f"gam_pdp_{group_name}.png"
    plt.savefig(f"{output_path}/{fname}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ PDP saved: {fname}")

for group_name, var_list in PDP_GROUPS.items():
    pdp_plot_group(group_name, var_list, SELECTED_VARS, gam_final, OUTPUT_PATH)

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── DONE ───")
print(f"n_splines          : {best_n_sp}")
print(f"lam                : {best_lam}")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"RMSE Train         : {metrics['train']['RMSE']:.6f}")
print(f"MAE  Train         : {metrics['train']['MAE']:.6f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"CV R²  (mean±std)  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"CV RMSE (mean±std) : {cv_rmse.mean():.6f} ± {cv_rmse.std():.6f}")
print(f"\nTop 5 features by permutation importance:")
for _, row in importance_df.head(5).iterrows():
    print(f"  {row['Variable']:<45} {row['Perm. Mean']:.4f} ± {row['Perm. Std']:.4f}")


# ─────────────────────────────────────────────
# SAVE FORECASTS FOR MCS
# ─────────────────────────────────────────────
MCS_PATH = "/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"
import os; os.makedirs(MCS_PATH, exist_ok=True)
np.save(f"{MCS_PATH}/y_pred_GAM.npy", y_pred_test)
print("✓ MCS forecasts saved (GAM).")