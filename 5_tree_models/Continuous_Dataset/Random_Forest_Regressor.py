import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
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
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/5_tree_models/Tables_Figures"

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
# NOTE ON SCALING
# Tree splits are based on ordinal thresholds, not feature magnitude.
# Random Forest is therefore invariant to monotonic feature scaling —
# no StandardScaler is applied (unlike Elastic Net / GAM).
# ─────────────────────────────────────────────

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
# n_estimators is fixed during the grid search: unlike tree depth /
# leaf size, the number of trees does not control model complexity in
# the overfitting sense (bagging variance reduction only improves,
# with diminishing returns, as more trees are added). It is therefore
# tuned separately as a one-dimensional stability check after the
# complexity-controlling hyperparameters are selected.
# ─────────────────────────────────────────────
N_ESTIMATORS_TUNE  = 300
N_ESTIMATORS_FINAL = 500

max_depth_grid        = [3, 5, 7, 10, None]
max_features_grid     = ["sqrt", "log2", 0.5, 1.0]
min_samples_leaf_grid = [1, 5, 10, 20]

best_rmse, best_depth, best_mf, best_msl = np.inf, None, None, None

print("Running 5-fold blocked CV for hyperparameter tuning ...")
for md in max_depth_grid:
    for mf in max_features_grid:
        for msl in min_samples_leaf_grid:
            fold_rmses = []
            for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
                m = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS_TUNE,
                    max_depth=md,
                    max_features=mf,
                    min_samples_leaf=msl,
                    n_jobs=-1,
                    random_state=42
                )
                m.fit(X_train[tr_idx], y_train[tr_idx])
                preds = m.predict(X_train[val_idx])
                fold_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))

            mean_rmse = np.mean(fold_rmses)
            print(f"  max_depth={md}, max_features={mf}, min_samples_leaf={msl} → CV RMSE={mean_rmse:.6f}")
            if mean_rmse < best_rmse:
                best_rmse  = mean_rmse
                best_depth = md
                best_mf    = mf
                best_msl   = msl

print(f"\nBest max_depth        : {best_depth}")
print(f"Best max_features     : {best_mf}")
print(f"Best min_samples_leaf : {best_msl}")
print(f"CV RMSE (best, n_estimators={N_ESTIMATORS_TUNE}): {best_rmse:.6f}")

# ─────────────────────────────────────────────
# REFIT FINAL MODEL ON FULL TRAINING SET
# n_estimators increased to N_ESTIMATORS_FINAL for a more stable
# (lower-variance) ensemble; complexity hyperparameters fixed from
# Stage 2. oob_score provides an additional, CV-independent check on
# generalization using the bootstrap out-of-bag samples.
# ─────────────────────────────────────────────
rf_final = RandomForestRegressor(
    n_estimators=N_ESTIMATORS_FINAL,
    max_depth=best_depth,
    max_features=best_mf,
    min_samples_leaf=best_msl,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf_final.fit(X_train, y_train)

# In-sample predictions (train)
y_pred_train = rf_final.predict(X_train)

# ─────────────────────────────────────────────
# STAGE 3 — ROLLING-ORIGIN EVALUATION (test set)
# Starting at the first test observation, the origin advances by one
# period each step, generating a series of one-step-ahead predictions.
# The model is NOT re-fitted; the ensemble is fixed from Stage 2.
# ─────────────────────────────────────────────
y_pred_rolling = np.empty(len(y_test))
for i in range(len(y_test)):
    y_pred_rolling[i] = rf_final.predict(X_test[i].reshape(1, -1))[0]

y_pred_test = y_pred_rolling

# ─────────────────────────────────────────────
# BLOCKED CV SCORES (for table reporting)
# Re-run with best hyperparameters (at N_ESTIMATORS_TUNE) for R² / RMSE per fold.
# ─────────────────────────────────────────────
cv_r2_scores   = []
cv_rmse_scores = []

for tr_idx, val_idx in blocked_cv_indices(n_train_obs, N_SPLITS):
    m_cv = RandomForestRegressor(
        n_estimators=N_ESTIMATORS_TUNE,
        max_depth=best_depth,
        max_features=best_mf,
        min_samples_leaf=best_msl,
        n_jobs=-1,
        random_state=42
    )
    m_cv.fit(X_train[tr_idx], y_train[tr_idx])
    preds = m_cv.predict(X_train[val_idx])

    cv_r2_scores.append(r2_score(y_train[val_idx], preds))
    cv_rmse_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))

cv_r2   = np.array(cv_r2_scores)
cv_rmse = np.array(cv_rmse_scores)

# ─────────────────────────────────────────────
# MODEL FIT METRICS
# OOS R² = 1 - SS_res / SS_tot  (applied to test set)
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
# TABLE 1: MODEL FIT
# ─────────────────────────────────────────────
sep = {"Metric": "", "Value": ""}

fit_rows = [
    {"Metric": "N (Train)",            "Value": str(n_train_n)},
    {"Metric": "N (Test)",             "Value": str(n_test_n)},
    {"Metric": "Features",             "Value": str(len(FEATURES))},
    {"Metric": "n_estimators",         "Value": str(N_ESTIMATORS_FINAL)},
    {"Metric": "max_depth",            "Value": str(best_depth)},
    {"Metric": "max_features",         "Value": str(best_mf)},
    {"Metric": "min_samples_leaf",     "Value": str(best_msl)},
    {"Metric": "OOB R²",               "Value": f"{rf_final.oob_score_:.4f}"},
    sep,
    {"Metric": "R² (Train)",           "Value": f"{metrics['train']['R²']:.4f}"},
    {"Metric": "RMSE (Train)",         "Value": f"{metrics['train']['RMSE']:.6f}"},
    {"Metric": "MAE (Train)",          "Value": f"{metrics['train']['MAE']:.6f}"},
    {"Metric": "OOS R² (Test)",        "Value": f"{metrics['test']['OOS R²']:.4f}"},
    {"Metric": "RMSE (Test)",          "Value": f"{metrics['test']['RMSE']:.6f}"},
    {"Metric": "MAE (Test)",           "Value": f"{metrics['test']['MAE']:.6f}"},
    {"Metric": "CV R² (mean)",         "Value": f"{cv_r2.mean():.4f}"},
    {"Metric": "CV R² (std)",          "Value": f"{cv_r2.std():.4f}"},
    {"Metric": "CV RMSE (mean)",       "Value": f"{cv_rmse.mean():.6f}"},
    {"Metric": "CV RMSE (std)",        "Value": f"{cv_rmse.std():.6f}"},
]

fit_df  = pd.DataFrame(fit_rows)
fit_tex = to_apa_latex_table(
    fit_df,
    "Random Forest Model Fit: TVL Log Return",
    "tab:rffit",
    note=(
        r"\textit{Note}: RandomForestRegressor (scikit-learn), bagged ensemble of regression trees. "
        r"Complexity hyperparameters (max\_depth, max\_features, min\_samples\_leaf) selected via "
        r"5-fold blocked cross-validation on the training set at n\_estimators$=300$; "
        r"the final ensemble is refit with n\_estimators$=500$ for variance stability. "
        r"OOB R$^2$ computed from bootstrap out-of-bag samples on the training set. "
        r"Test-set forecasts obtained via rolling-origin evaluation (one-step-ahead); "
        r"the ensemble is fixed after Stage 2 and not re-estimated during rolling evaluation. "
        r"OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    ),
)

with open(f"{OUTPUT_PATH}/rf_fit.tex", "w") as f:
    f.write(fit_tex)
print("✓ Fit table saved.")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# Gini importance (mean decrease in variance, training set, biased
# toward high-cardinality / correlated features) reported alongside
# permutation importance (computed on the held-out test set, n_repeats
# = 10), which is more reliable but does not bias the OOS evaluation
# since hyperparameters were already fixed in Stage 2.
# ─────────────────────────────────────────────
gini_importance = rf_final.feature_importances_

perm_result = permutation_importance(
    rf_final, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

importance_df = pd.DataFrame({
    "Variable":       FEATURES,
    "Gini":           gini_importance,
    "Perm. Mean":     perm_result.importances_mean,
    "Perm. Std":      perm_result.importances_std,
}).sort_values("Perm. Mean", ascending=False).reset_index(drop=True)

imp_rows = []
for _, row in importance_df.iterrows():
    imp_rows.append({
        "Variable":             row["Variable"],
        "Gini Importance":      f"{row['Gini']:.4f}",
        "Perm. Importance":     f"{row['Perm. Mean']:.4f}",
        "Perm. Std.":           f"{row['Perm. Std']:.4f}",
    })

imp_df_tex_input = pd.DataFrame(imp_rows)
imp_tex = to_apa_latex_table(
    imp_df_tex_input,
    "Random Forest: Feature Importance (Sorted by Permutation Importance)",
    "tab:rfimportance",
    note=(
        r"\textit{Note}: Gini importance = mean decrease in variance (training set), "
        r"can be biased toward correlated or high-cardinality features. "
        r"Permutation importance = mean decrease in $R^2$ over 10 repeats when a "
        r"feature is randomly shuffled, computed on the held-out test set. "
        r"Sorted descending by permutation importance."
    )
)

with open(f"{OUTPUT_PATH}/rf_importance.tex", "w") as f:
    f.write(imp_tex)
print("✓ Feature importance table saved.")

# ─────────────────────────────────────────────
# PLOT: ACTUAL vs PREDICTED (test set, rolling-origin)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(dates_test, y_test,      color="black",   linewidth=0.8, label="Actual")
ax.plot(dates_test, y_pred_test, color="#B22222", linewidth=0.8, linestyle="--", label="Predicted")

ax.set_title("Random Forest: Actual vs. Predicted TVL Log Return (Test Set)",
             fontstyle="italic", fontsize=12, pad=4)
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

fig.suptitle("Random Forest Regression: Out-of-Sample Fit",
             fontstyle="italic", fontsize=13, y=1.01)
fig.text(0.0, -0.02,
         "Note. Dashed red line = Random Forest one-step-ahead predictions (rolling-origin). "
         "Solid black line = realized TVL log returns. "
         "Test period covers the final 20% of the sample (chronological split).",
         fontsize=10, ha="left", wrap=True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/rf_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Actual vs. Predicted plot saved.")

# ─────────────────────────────────────────────
# PLOT: FEATURE IMPORTANCE (permutation, top 20)
# ─────────────────────────────────────────────
top_n = 20
plot_df = importance_df.head(top_n).sort_values("Perm. Mean")

fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
ax.barh(plot_df["Variable"], plot_df["Perm. Mean"],
        xerr=plot_df["Perm. Std"], color="#2C3E50", edgecolor="none",
        error_kw={"elinewidth": 0.8, "ecolor": "#888888", "capsize": 2})
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Permutation Importance (Mean Decrease in $R^2$)", fontsize=11)
ax.set_title("Random Forest: Top 20 Features by Permutation Importance",
             fontstyle="italic", fontsize=12, pad=4)
ax.tick_params(axis="both", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Random Forest Variable Importance", fontstyle="italic", fontsize=13, y=1.01)
fig.text(0.0, -0.02,
         "Note. Permutation importance computed on the held-out test set (10 repeats); "
         "error bars show one standard deviation across repeats. "
         f"Showing top {top_n} of {len(FEATURES)} features.",
         fontsize=10, ha="left", wrap=True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/rf_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Feature importance plot saved.")

# ─────────────────────────────────────────────
# PLOT: PARTIAL DEPENDENCE (grouped by variable category)
# Mirrors the GAM PDP plots (same groups) for direct cross-model
# comparison: smooth spline effects (GAM) vs. step-function /
# threshold effects (Random Forest) for the same variables.
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

def pdp_plot_group(group_name, var_list, all_features, model, X_data, output_path):
    vars_to_plot = [v for v in var_list if v in all_features]
    if not vars_to_plot:
        return

    n_vars = len(vars_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, var in enumerate(vars_to_plot):
        feat_idx = all_features.index(var)
        ax = axes[idx]

        pdp_result = partial_dependence(
            model, X_data, features=[feat_idx], kind="average", grid_resolution=50
        )
        grid_vals = pdp_result["grid_values"][0]
        avg_vals  = pdp_result["average"][0]

        ax.plot(grid_vals, avg_vals, color="black", linewidth=1.0, drawstyle="steps-mid")
        ax.set_title(var.replace("_", " "), fontstyle="italic", fontsize=9, pad=3)
        ax.set_xlabel("Feature Value", fontsize=8)
        ax.set_ylabel("Partial Effect", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    title = group_name.replace("_", " ")
    fig.suptitle(f"Random Forest Partial Dependence Plots: {title}",
                 fontstyle="italic", fontsize=13, y=1.01)
    fig.text(0.0, -0.01,
             "Note. Partial dependence (average prediction over training set) computed via "
             "the recursion method. Step-shaped curves reflect the piecewise-constant nature "
             "of tree-based partitioning, in contrast to the smooth GAM splines.",
             fontsize=10, ha="left", wrap=True)

    plt.tight_layout()
    fname = f"rf_pdp_{group_name}.png"
    plt.savefig(f"{output_path}/{fname}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ PDP saved: {fname}")

for group_name, var_list in PDP_GROUPS.items():
    pdp_plot_group(group_name, var_list, FEATURES, rf_final, X_train, OUTPUT_PATH)

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── DONE ───")
print(f"max_depth          : {best_depth}")
print(f"max_features       : {best_mf}")
print(f"min_samples_leaf   : {best_msl}")
print(f"OOB R²             : {rf_final.oob_score_:.4f}")
print(f"R²  Train          : {metrics['train']['R²']:.4f}")
print(f"RMSE Train         : {metrics['train']['RMSE']:.6f}")
print(f"MAE  Train         : {metrics['train']['MAE']:.6f}")
print(f"OOS R² Test        : {metrics['test']['OOS R²']:.4f}")
print(f"RMSE Test          : {metrics['test']['RMSE']:.6f}")
print(f"MAE  Test          : {metrics['test']['MAE']:.6f}")
print(f"CV R²  (mean±std)  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"CV RMSE (mean±std) : {cv_rmse.mean():.6f} ± {cv_rmse.std():.6f}")