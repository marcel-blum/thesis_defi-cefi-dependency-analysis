import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
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
MCS_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/9_Model_Comparison/Model_Confidence_Set/MCS_Tables_Figures"
os.makedirs(MCS_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD FORECAST ARRAYS
# Each model script saves y_pred_<MODEL>.npy and a shared y_test.npy.
# ARX and TFT also save their own y_test for consistency checking.
# ─────────────────────────────────────────────
y_test = np.load(f"{MCS_PATH}/y_test.npy")

model_names = ["MLR", "ENet", "GAM", "XGB", "SVR", "ARX", "TFT"]
preds = {}
for m in model_names:
    preds[m] = np.load(f"{MCS_PATH}/y_pred_{m}.npy")

# ── Consistency checks for ARX and TFT ──────
for m in ["ARX", "TFT"]:
    y_test_m = np.load(f"{MCS_PATH}/y_test_{m}.npy")
    if not np.allclose(y_test, y_test_m, atol=1e-10):
        raise ValueError(
            f"y_test mismatch for {m}! "
            f"Check that all models use the same 80/20 split and the same target column."
        )
print("✓ y_test consistency check passed for ARX and TFT.")

# ── Sanity: equal length ─────────────────────
n_test = len(y_test)
for m, p in preds.items():
    assert len(p) == n_test, f"Length mismatch for {m}: {len(p)} vs {n_test}"
print(f"✓ All forecast arrays loaded. n_test = {n_test}")

# ─────────────────────────────────────────────
# SQUARED LOSS DIFFERENTIALS
# d_ij(t) = e_i(t)^2 - e_j(t)^2
# Hansen, Lunde & Nason (2011) use squared-error loss by default.
# ─────────────────────────────────────────────
losses = {}
for m in model_names:
    losses[m] = (y_test - preds[m]) ** 2   # squared errors, shape (n_test,)

# ─────────────────────────────────────────────
# MCS IMPLEMENTATION
# Range statistic T_R (Hansen et al., 2011, eq. 14):
#   T_R = max_{i,j in M} |t_ij|
# where t_ij = d_ij_bar / sqrt(Var(d_ij_bar))
# Var estimated via Newey-West HAC (accounts for serial correlation
# in squared-error loss differentials — common in financial return series).
# Elimination rule: remove the model i* with the largest mean loss
# among models involved in the most extreme pairwise comparison.
# Bootstrap p-values via stationary block bootstrap (B=5000 replications,
# block length l=sqrt(n_test) as a practical default).
# ─────────────────────────────────────────────

def newey_west_variance(x, max_lag=None):
    """
    Newey-West HAC estimator of Var(mean(x)).
    max_lag defaults to floor(4*(n/100)^(2/9)) following the
    Andrews (1991) rule commonly used in forecast evaluation.
    """
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    x_dm = x - x.mean()
    gamma_0 = np.dot(x_dm, x_dm) / n
    s2 = gamma_0
    for lag in range(1, max_lag + 1):
        gamma_l = np.dot(x_dm[lag:], x_dm[:-lag]) / n
        w = 1 - lag / (max_lag + 1)   # Bartlett kernel
        s2 += 2 * w * gamma_l
    return max(s2, 1e-16) / n         # Var(mean) = S^2 / n

def t_statistic(d_ij):
    """t-stat for H0: E[d_ij] = 0."""
    d_bar = d_ij.mean()
    var   = newey_west_variance(d_ij)
    return d_bar / np.sqrt(var)

def range_statistic(model_set, losses):
    """T_R = max_{i≠j in M} |t_ij|."""
    m_list = list(model_set)
    T_R = 0.0
    worst_pair = (None, None)
    for i in range(len(m_list)):
        for j in range(i + 1, len(m_list)):
            mi, mj = m_list[i], m_list[j]
            d_ij   = losses[mi] - losses[mj]
            t      = abs(t_statistic(d_ij))
            if t > T_R:
                T_R = t
                worst_pair = (mi, mj)
    return T_R, worst_pair

def eliminate_model(model_set, losses):
    """
    Identify the model to eliminate: the one with the highest mean
    squared error among all models in the current set.
    This corresponds to the 'max-loss' elimination rule.
    """
    mean_losses = {m: losses[m].mean() for m in model_set}
    return max(mean_losses, key=mean_losses.get)

def block_bootstrap_pvalue(model_set, losses, T_obs, B=5000, seed=42):
    """
    Stationary block bootstrap p-value for T_R.
    Block length l = floor(sqrt(n_test)).
    """
    rng    = np.random.default_rng(seed)
    n      = len(y_test)
    l      = max(1, int(np.floor(np.sqrt(n))))
    m_list = list(model_set)

    # Loss differentials matrix: shape (n_pairs, n)
    pairs    = [(m_list[i], m_list[j])
                for i in range(len(m_list))
                for j in range(i + 1, len(m_list))]
    d_matrix = np.array([losses[mi] - losses[mj] for mi, mj in pairs])  # (P, n)

    count = 0
    for _ in range(B):
        # Draw block-bootstrap indices
        starts = rng.integers(0, n, size=int(np.ceil(n / l)))
        idx    = np.concatenate([np.arange(s, min(s + l, n)) for s in starts])[:n]

        T_boot = 0.0
        for d_ij in d_matrix:
            d_boot  = d_ij[idx]
            # Re-center: subtract observed mean (ensures correct null distribution)
            d_boot  = d_boot - d_ij.mean()
            t_boot  = abs(d_boot.mean() / np.sqrt(max(newey_west_variance(d_boot), 1e-16)))
            if t_boot > T_boot:
                T_boot = t_boot
        if T_boot >= T_obs:
            count += 1

    return count / B

# ─────────────────────────────────────────────
# MCS ALGORITHM — SEQUENTIAL ELIMINATION
# ─────────────────────────────────────────────
ALPHA   = 0.10          # significance level (90% MCS — standard in literature)
B       = 5000          # bootstrap replications

print(f"\nRunning MCS (α={ALPHA}, B={B} bootstrap replications) ...")

model_set   = set(model_names)
mcs_results = []          # records (eliminated_model, T_R, p_value, step)
step        = 0

while len(model_set) > 1:
    step += 1
    T_obs, worst_pair = range_statistic(model_set, losses)
    p_val = block_bootstrap_pvalue(model_set, losses, T_obs, B=B)

    print(f"  Step {step}: M={sorted(model_set)}, T_R={T_obs:.4f}, p={p_val:.4f}")

    if p_val < ALPHA:
        # Reject H0 — eliminate the worst model
        eliminated = eliminate_model(model_set, losses)
        mcs_results.append({
            "Step":       step,
            "Model Set":  ", ".join(sorted(model_set)),
            "T_R":        T_obs,
            "p-Value":    p_val,
            "Eliminated": eliminated,
            "In MCS*":    "No"
        })
        print(f"    → Eliminated: {eliminated} (p={p_val:.4f} < {ALPHA})")
        model_set.remove(eliminated)
    else:
        # Fail to reject — remaining models form the MCS*
        for m in sorted(model_set):
            mcs_results.append({
                "Step":       step,
                "Model Set":  ", ".join(sorted(model_set)),
                "T_R":        round(T_obs, 4),
                "p-Value":    round(p_val, 4),
                "Eliminated": "—",
                "In MCS*":    "Yes"
            })
        print(f"    → MCS* = {sorted(model_set)} (p={p_val:.4f} ≥ {ALPHA})")
        break

# If only one model remains after all eliminations
if len(model_set) == 1:
    m = list(model_set)[0]
    mcs_results.append({
        "Step":       step,
        "Model Set":  m,
        "T_R":        0.0,
        "p-Value":    1.0,
        "Eliminated": "—",
        "In MCS*":    "Yes"
    })

mcs_star = [r["Model Set"] for r in mcs_results if r["In MCS*"] == "Yes"]
mcs_star_models = sorted(model_set)
print(f"\n✓ MCS* (at α={ALPHA}): {mcs_star_models}")

# ─────────────────────────────────────────────
# SUMMARY TABLE — PER MODEL
# Columns: Model | RMSE | MAE | OOS R² | Mean Sq. Loss | In MCS*
# ─────────────────────────────────────────────
def oos_r2(y_true, y_pred, y_train_mean):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_train_mean) ** 2)

def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

# y_train mean: reconstruct from dataset (same split logic as all models)
INPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
df_raw     = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
TARGET     = "Global_TVL_USD_log_return"
y_all      = df_raw[TARGET].dropna().values
split_idx  = int(len(y_all) * 0.80)
y_train_mean = y_all[:split_idx].mean()

summary_rows = []
for m in model_names:
    p       = preds[m]
    rmse    = np.sqrt(np.mean((y_test - p) ** 2))
    mae     = np.mean(np.abs(y_test - p))
    r2      = oos_r2(y_test, p, y_train_mean)
    msl     = losses[m].mean()
    in_mcs  = "\\checkmark" if m in mcs_star_models else "—"
    summary_rows.append({
        "Model":          m,
        "RMSE":           fmt4(rmse),
        "MAE":            fmt4(mae),
        "OOS $R^2$":      fmt4(r2),
        "Mean Sq. Loss":  fmt4(msl),
        "In MCS$^*$":     in_mcs
    })

summary_df  = pd.DataFrame(summary_rows)
summary_tex = to_apa_latex_table(
    summary_df,
    note=(
        r"\textit{Note}: MCS computed following \cite{HansenLundeNason2011}. "
        r"Range statistic $T_R$ with block bootstrap $p$-values "
        r"($B = 5{,}000$ replications; block length $l = \lfloor\sqrt{T}\rfloor$; "
        r"Newey-West HAC variance estimator). "
        r"Elimination at significance level $\alpha = 0.10$; "
        r"models marked \checkmark~belong to the superior model confidence set MCS$^*$. "
        r"Loss function: squared forecast error. "
        r"OOS $R^2 = 1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$, "
        r"where $\text{SS}_{\text{tot}}$ is computed relative to the in-sample training mean. "
        r"All forecasts obtained via rolling-origin evaluation (one-step-ahead) on the held-out test set."
    )
)
with open(f"{MCS_PATH}/mcs_summary.tex", "w") as f:
    f.write(summary_tex)
print("✓ MCS summary table saved.")

# ─────────────────────────────────────────────
# ELIMINATION PATH TABLE
# ─────────────────────────────────────────────
path_rows = []
for r in mcs_results:
    if r["Eliminated"] != "—":
        path_rows.append({
            "Step":        str(r["Step"]),
            "Active Set":  r["Model Set"],
            "$T_R$":       fmt4(r['T_R']),
            "$p$-Value":   fmt4(r['p-Value']),
            "Eliminated":  r["Eliminated"]
        })

if path_rows:
    path_df  = pd.DataFrame(path_rows)
    path_tex = to_apa_latex_table(
        path_df,
        note=(
            r"\textit{Note}: Sequential elimination procedure of the Model Confidence Set "
            r"At each step, the range statistic $T_R$ is computed "
            r"over the active model set; if the bootstrap $p$-value falls below $\alpha = 0.10$, "
            r"the model with the highest mean squared loss is eliminated."
        )
    )
    with open(f"{MCS_PATH}/mcs_elimination_path.tex", "w") as f:
        f.write(path_tex)
    print("✓ MCS elimination path table saved.")

# ─────────────────────────────────────────────
# PLOT: MEAN SQUARED LOSS PER MODEL
# Bar chart; MCS* members highlighted in black, eliminated in grey.
# ─────────────────────────────────────────────
mean_losses_vals = [losses[m].mean() for m in model_names]
colors           = ["black" if m in mcs_star_models else "#AAAAAA" for m in model_names]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(model_names, mean_losses_vals, color=colors, edgecolor="black", linewidth=0.7)

ax.set_ylabel("Mean Squared Loss", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.tick_params(axis="both", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))

# Legend
patch_in  = mpatches.Patch(color="black",   label=r"In MCS$^*$")
patch_out = mpatches.Patch(color="#AAAAAA", label=r"Eliminated")
ax.legend(handles=[patch_in, patch_out], fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(f"{MCS_PATH}/mcs_loss_barplot.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ MCS loss bar plot saved.")

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n─── MCS DONE ───")
print(f"α                  : {ALPHA}")
print(f"Bootstrap reps     : {B}")
print(f"n_test             : {n_test}")
print(f"MCS* members       : {mcs_star_models}")
print(f"Eliminated models  : {[m for m in model_names if m not in mcs_star_models]}")