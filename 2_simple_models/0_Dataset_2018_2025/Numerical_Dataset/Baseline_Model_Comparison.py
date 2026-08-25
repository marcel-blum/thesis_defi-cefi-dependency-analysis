import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# load table formatting helper
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

# --- paths ---
output_dir = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/2_simple_models/0_Dataset_2018_2025/Tables_Figures"

# ─────────────────────────────────────────────
# HELPER: 4-DECIMAL FORMATTING
# ─────────────────────────────────────────────
def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"


def fit_train_test(df, target, split=0.80):
    """
    80/20 chronological split, HC3-OLS on train, rolling-origin one-step-ahead
    forecasts on test — mirrors the methodology used in the individual MLR
    scripts (Multiple_Linear_Regression_2018_2025.py / _2020_2025.py), so this
    comparison reports genuine out-of-sample metrics rather than full-sample
    (in-sample-only) fit statistics.
    """
    features = [c for c in df.columns if c not in ["date", target]]
    df_model = df[["date", target] + features].dropna().reset_index(drop=True)

    X = df_model[features].values
    y = df_model[target].values
    dates = df_model["date"].values

    split_idx = int(len(df_model) * split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    X_sm_train = sm.add_constant(X_train, has_constant="add")
    ols_result = sm.OLS(y_train, X_sm_train).fit(cov_type="HC3")

    y_pred_train = ols_result.fittedvalues
    r2_train  = ols_result.rsquared
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train  = mean_absolute_error(y_train, y_pred_train)

    # rolling-origin one-step-ahead forecasts (fixed training model applied
    # sequentially to each test observation — no lagged DV in a plain MLR)
    X_sm_test = sm.add_constant(X_test, has_constant="add")
    y_pred_test = np.empty(len(y_test))
    for i in range(len(y_test)):
        y_pred_test[i] = ols_result.predict(X_sm_test[i].reshape(1, -1))[0]

    oos_r2 = 1 - np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_train)) ** 2)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test  = mean_absolute_error(y_test, y_pred_test)

    kpis = {
        "R² (Train)":    r2_train,
        "RMSE (Train)":  rmse_train,
        "MAE (Train)":   mae_train,
        "OOS R² (Test)": oos_r2,
        "RMSE (Test)":   rmse_test,
        "MAE (Test)":    mae_test,
    }
    return kpis, dates_test, y_test, y_pred_test

# --- load datasets ---
df_long = pd.read_csv(
    "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Final_Dataset_numTVL_2018_2025.csv",
    parse_dates=["date"]
)
df_short = pd.read_csv(
    "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv",
    parse_dates=["date"]
)

# --- fit models (80/20 split, HC3-OLS train, rolling-origin test) ---
kpis_long,  dates_test_long,  y_test_long,  pred_test_long  = fit_train_test(df_long,  "Global_TVL_USD_log_return_w")
kpis_short, dates_test_short, y_test_short, pred_test_short = fit_train_test(df_short, "Global_TVL_USD_log_return")

# --- KPI comparison table (Train + Test/OOS metrics, both periods) ---
merged = pd.DataFrame({
    "Metric":      list(kpis_long.keys()),
    "2018 - 2025": [fmt4(v) for v in kpis_long.values()],
    "2020 - 2025": [fmt4(kpis_short[m]) for m in kpis_long.keys()],
})

# NOTE: to_apa_latex_table() returns ONLY the tabular fragment + note.
# Caption and label are supplied in-text via the \thesistable{...}{...}{...}{...} macro.
mlrkpi_comparison = to_apa_latex_table(
    merged,
    note=(
        r"\textit{Note}: 80/20 chronological train/test split. HC3 "
        r"heteroskedasticity-robust standard errors on the training set. "
        r"Test-set forecasts obtained via rolling-origin evaluation "
        r"(one-step-ahead). OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$."
    )
)

output_path = os.path.join(output_dir, "mlrkpi_time_period_comparison.tex")
with open(output_path, "w") as f:
    f.write(mlrkpi_comparison)

## visualization ##
# --- combined figure: OOS Actual vs. Predicted (test set), both periods ---
# NOTE: content only — no in-image suptitle/note (those belong in the LaTeX
# \caption / \fignote). Per-panel titles are kept since they distinguish the
# two dataset panels (multi-panel figure). Line spans the combined range of
# BOTH actual and predicted values, axes forced to equal aspect, so the y=x
# "perfect fit" reference line is a true 45-degree diagonal.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

panels = [
    (pred_test_long,  y_test_long,  "2018 - 2025 (Long Dataset)"),
    (pred_test_short, y_test_short, "2020 - 2025 (Short Dataset)")
]

for ax, (pred, actual, title) in zip(axes, panels):
    ax.scatter(pred, actual, s=5, alpha=0.3, color='black')

    lo = min(pred.min(), actual.min())
    hi = max(pred.max(), actual.max())
    lims = [lo, hi]
    ax.plot(lims, lims, color='red', linewidth=1, linestyle='--', label='y = x (perfect fit)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Predicted Values (Test Set)", fontsize=11)
    ax.set_ylabel("Actual Values (Test Set)", fontsize=11)
    ax.set_title(title, fontstyle='italic', fontsize=12, pad=4)
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(
    "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/2_simple_models/0_Dataset_2018_2025/Tables_Figures/mlr_actual_vs_fitted_comparison.png",
    dpi=300, bbox_inches='tight'
)
plt.close()
