import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, partial_dependence
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
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/6_support_vector_machines/Tables_Figures"

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
# SVR with RBF kernel relies on Euclidean distances in feature space.
# Features must be standardized to prevent high-magnitude variables
# from dominating the kernel computation.
# Scaler is fit exclusively on training data to prevent data leakage.
# ─────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

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
# 5-fold blocked CV on training set, parallelized across hyperparameter
# combinations via joblib (process-based, n_jobs=-2 = all logical cores
# minus one, keeping the machine responsive for other work).
# libsvm (sklearn SVR backend) is single-threaded internally; parallelism
# is therefore applied at the outer grid-search level, which is the
# efficient choice for this model class.
#
# Hyperparameters tuned:
#   C       — regularization: trades margin width against training error
#   epsilon — width of the epsilon-insensitive tube (residuals within
#             epsilon are not penalized)
#   gamma   — RBF kernel bandwidth: controls the influence radius of
#             individual support vectors in feature space
# ─────────────────────────────────────────────
C_grid       = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
epsilon_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
gamma_grid   = [0.0001, 0.0005, 0.001, 0.005, 0.01, "scale"]

param_grid = [
    (c, eps, gam)
    for c   in C_grid
    for eps in epsilon_grid
    for gam in gamma_grid
]

def evaluate_combo(params):
    c, eps, gam = params
    fold_rmses, fold_r2s = [], []
    for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
        # Scale within each fold to prevent leakage across blocks
        sc_fold   = StandardScaler()
        Xtr_fold  = sc_fold.fit_transform(X_train[tr_idx])
        Xval_fold = sc_fold.transform(X_train[val_idx])

        m = SVR(kernel="rbf", C=c, epsilon=eps, gamma=gam)
        m.fit(Xtr_fold, y_train[tr_idx])
        preds = m.predict(Xval_fold)

        fold_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))
        fold_r2s.append(r2_score(y_train[val_idx], preds))

    return {
        "params":    params,
        "rmse_mean": float(np.mean(fold_rmses)),
        "rmse_std":  float(np.std(fold_rmses)),
        "r2_mean":   float(np.mean(fold_r2s)),
        "r2_std":    float(np.std(fold_r2s)),
    }

print(f"Running 5-fold blocked CV for hyperparameter tuning over {len(param_grid)} "
      f"combinations (parallelized, n_jobs=-2) ...")

cv_results = Parallel(n_jobs=-2, verbose=10)(
    delayed(evaluate_combo)(p) for p in param_grid
)

best_result  = min(cv_results, key=lambda r: r["rmse_mean"])
best_C, best_epsilon, best_gamma = best_result["params"]

cv_rmse_mean, cv_rmse_std = best_result["rmse_mean"], best_result["rmse_std"]
cv_r2_mean,   cv_r2_std   = best_result["r2_mean"],   best_result["r2_std"]

print(f"\nBest C       : {best_C}")
print(f"Best epsilon : {best_epsilon}")
print(f"Best gamma   : {best_gamma}")
print(f"CV RMSE (best): {cv_rmse_mean:.6f} ± {cv_rmse_std:.6f}")
print(f"CV R²   (best): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

# ─────────────────────────────────────────────
# REFIT FINAL MODEL ON FULL TRAINING SET
# ─────────────────────────────────────────────
svr_final = SVR(kernel="rbf", C=best_C, epsilon=best_epsilon, gamma=best_gamma)
svr_final.fit(X_train_sc, y_train)

# In-sample predictions (train)
y_pred_train = svr_final.predict(X_train_sc)

# ─────────────────────────────────────────────
# STAGE 3 — ROLLING-ORIGIN EVALUATION (test set)
# Starting at the first test observation, the origin advances by one
# period each step, generating a series of one-step-ahead predictions.
# The model is NOT re-fitted; the SVR is fixed from Stage 2.
# ─────────────────────────────────────────────
y_pred_rolling = np.empty(len(y_test))
for i in range(len(y_test)):
    y_pred_rolling[i] = svr_final.predict(X_test_sc[i].reshape(1, -1))[0]

y_pred_test = y_pred_rolling

# ─────────────────────────────────────────────
# MODEL FIT METRICS
# ─────────────────────────────────────────────
n_train_n = len(y_train)
n_test_n  = len(y_test)

oos_r2 = 1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_train))**2)

def adj_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

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
# TABLE: MODEL FIT
# ─────────────────────────────────────────────
sep = {"Metric": "", "Value": ""}

fit_rows = [
    {"Metric": "N (Train)",      "Value": str(n_train_n)},
    {"Metric": "N (Test)",       "Value": str(n_test_n)},
    {"Metric": "Features",       "Value": str(len(FEATURES))},
    {"Metric": "Kernel",         "Value": "RBF"},
    {"Metric": "C",              "Value": str(best_C)},
    {"Metric": "epsilon",        "Value": str(best_epsilon)},
    {"Metric": "gamma",          "Value": str(best_gamma)},
    sep,
    {"Metric": "R² (Train)",     "Value": fmt4(metrics['train']['R²'])},
    {"Metric": "RMSE (Train)",   "Value": fmt4(metrics['train']['RMSE'])},
    {"Metric": "MAE (Train)",    "Value": fmt4(metrics['train']['MAE'])},
    {"Metric": "OOS R² (Test)",  "Value": fmt4(metrics['test']['OOS R²'])},
    {"Metric": "RMSE (Test)",    "Value": fmt4(metrics['test']['RMSE'])},
    {"Metric": "MAE (Test)",     "Value": fmt4(metrics['test']['MAE'])},
    {"Metric": "CV R² (mean)",   "Value": fmt4(cv_r2_mean)},
    {"Metric": "CV R² (std)",    "Value": fmt4(cv_r2_std)},
    {"Metric": "CV RMSE (mean)", "Value": fmt4(cv_rmse_mean)},
    {"Metric": "CV RMSE (std)",  "Value": fmt4(cv_rmse_std)},
]

fit_df  = pd.DataFrame(fit_rows)
fit_tex = to_apa_latex_table(
    fit_df,
    note=(
        r"\textit{Note}: Support Vector Regression with RBF kernel (sklearn SVR, libsvm backend). "
        r"Features standardized prior to estimation; scaler fitted exclusively on the training set. "
        r"Hyperparameters selected via 5-fold blocked cross-validation on the training set, "
        r"parallelized across all candidate combinations (joblib, n\_jobs=-2). "
        r"Test-set forecasts obtained via rolling-origin evaluation (one-step-ahead); "
        r"the model is fixed after Stage 2 and not re-estimated during rolling evaluation. "
        r"OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    ),
)

with open(f"{OUTPUT_PATH}/svr_fit.tex", "w") as f:
    f.write(fit_tex)
print("✓ Fit table saved.")

# ─────────────────────────────────────────────
# PERMUTATION IMPORTANCE
# SVR has no native feature importances (no coefficients in the
# original feature space for RBF kernel, no tree-based gain).
# Permutation importance is the model-agnostic alternative:
# mean decrease in R² over n_repeats=10 random shuffles of each
# feature, computed on the held-out test set.
# n_jobs=-2: parallelized across features and repeats.
# ─────────────────────────────────────────────
perm_result = permutation_importance(
    svr_final, X_test_sc, y_test,
    n_repeats=10, random_state=42, n_jobs=-2,
    scoring="r2"
)

importance_df = pd.DataFrame({
    "Variable":   FEATURES,
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
        r"SVR with RBF kernel does not provide native feature importances; "
        r"permutation importance is therefore the sole attribution metric reported. "
        r"Sorted descending by permutation importance."
    )
)

with open(f"{OUTPUT_PATH}/svr_importance.tex", "w") as f:
    f.write(imp_tex)
print("✓ Feature importance table saved.")

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
plt.savefig(f"{OUTPUT_PATH}/svr_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Actual vs. Predicted plot saved.")

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
plt.savefig(f"{OUTPUT_PATH}/svr_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Permutation importance plot saved.")

# ─────────────────────────────────────────────
# PLOT: PARTIAL DEPENDENCE (grouped by variable category)
# Same groups as GAM / XGBoost for direct cross-model comparison.
# sklearn's partial_dependence with method="auto" uses the brute-force
# approach for SVR (no recursion shortcut available), re-predicting
# over the training set for each grid point.
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
        "EZ_Rate_Hike",  "EZ_Rate_Cut",
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

def pdp_plot_group(group_name, var_list, all_features, model, X_data, output_path):
    vars_to_plot = [v for v in var_list if v in all_features]
    if not vars_to_plot:
        return

    n_vars = len(vars_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, var in enumerate(vars_to_plot):
        feat_idx = all_features.index(var)
        ax = axes[idx]

        pdp_result = partial_dependence(
            model, X_data, features=[feat_idx], kind="average",
            grid_resolution=30, method="auto"
        )
        grid_vals = pdp_result["grid_values"][0]
        avg_vals  = pdp_result["average"][0]

        ax.plot(grid_vals, avg_vals, color="black", linewidth=1.0)
        ax.set_title(var.replace("_", " "), fontstyle="italic", fontsize=15, pad=6)
        ax.set_xlabel("Feature Value (Standardized)", fontsize=13)
        ax.set_ylabel("Partial Effect", fontsize=13)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.tick_params(labelsize=11)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fname = f"svr_pdp_{group_name}.png"
    plt.savefig(f"{output_path}/{fname}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ PDP saved: {fname}")

for group_name, var_list in PDP_GROUPS.items():
    pdp_plot_group(group_name, var_list, FEATURES, svr_final, X_train_sc, OUTPUT_PATH)

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── DONE ───")
print(f"Kernel             : RBF")
print(f"C                  : {best_C}")
print(f"epsilon            : {best_epsilon}")
print(f"gamma              : {best_gamma}")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"RMSE Train         : {metrics['train']['RMSE']:.6f}")
print(f"MAE  Train         : {metrics['train']['MAE']:.6f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"CV R²  (mean±std)  : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
print(f"CV RMSE (mean±std) : {cv_rmse_mean:.6f} ± {cv_rmse_std:.6f}")


# ─────────────────────────────────────────────
# SAVE FORECASTS FOR MCS
# ─────────────────────────────────────────────
MCS_PATH = "/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"
import os; os.makedirs(MCS_PATH, exist_ok=True)
np.save(f"{MCS_PATH}/y_pred_SVR.npy", y_pred_test)
print("✓ MCS forecasts saved (SVR).")