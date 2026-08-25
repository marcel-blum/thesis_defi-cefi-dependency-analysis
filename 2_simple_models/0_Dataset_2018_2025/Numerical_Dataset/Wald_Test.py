import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers")
from table_format import to_apa_latex_table

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

def format_pval(p):
    return r"$<$0.0001" if p < 0.0001 else f"{p:.4f}"

## Structural Break Test (Chow / Robust Wald, HC3) - Long Dataset (2018-2025)
## Split date: 2020-08-12

df = pd.read_csv(
    "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Final_Dataset_numTVL_2018_2025.csv",
    parse_dates=["date"]
)

y = df["Global_TVL_USD_log_return_w"].values
x = df.drop(columns=["date", "Global_TVL_USD_log_return_w"])
X_base = x.values
n, k = X_base.shape
var_names = list(x.columns)

SPLIT_DATE = "2020-08-12"
D = (df["date"] >= SPLIT_DATE).astype(int).values

# --- Identify variables with no within-regime variation ---
TOL = 1e-12
valid_cols = []
dropped_vars = []
for i in range(k):
    var_pre = np.var(X_base[D == 0, i])
    var_post = np.var(X_base[D == 1, i])
    if var_pre > TOL and var_post > TOL:
        valid_cols.append(i)
    else:
        dropped_vars.append(var_names[i])

print(f"Excluded from interaction terms ({len(dropped_vars)}): {dropped_vars}")

# --- Unrestricted model ---
X_interactions = D[:, None] * X_base[:, valid_cols]
X_design = np.column_stack([np.ones(n), D, X_base, X_interactions])

model = sm.OLS(y, X_design).fit(cov_type="HC3")

# --- Restriction matrix: D and remaining D*X_i = 0 ---
n_params = X_design.shape[1]
n_interactions = len(valid_cols)
restriction_idx = [1] + list(range(n_params - n_interactions, n_params))
n_restrictions = len(restriction_idx)

R = np.zeros((n_restrictions, n_params))
for i, idx in enumerate(restriction_idx):
    R[i, idx] = 1

rank_check = np.linalg.matrix_rank(R @ model.cov_params() @ R.T)
print(f"Restrictions: {n_restrictions}, Rank: {rank_check}")

wald = model.wald_test(R, scalar=True)

stat = float(wald.statistic)
p_value = float(wald.pvalue)
dof = int(wald.df_denom)
p_display = format_pval(p_value)

decision = "Reject H0: structural break detected" if p_value < 0.05 else "Fail to reject H0: no structural break"

# --- Results Table ---
stars = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""

results_df = pd.DataFrame({
    "Test Statistic": [f"$\\chi^2$ = {fmt4(stat)}{stars}"],
    "df": [str(dof)],
    "p-value": [p_display]
})

# NOTE: to_apa_latex_table() returns ONLY the tabular fragment + note.
# Caption and label are supplied in-text via the \thesistable{...}{...}{...}{...} macro.
results_tex = to_apa_latex_table(
    results_df,
    note=(
        r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1. "
        r"H$_0$: coefficients on the post-break dummy and interaction terms are jointly zero. "
        + str(len(dropped_vars)) + r" dummy variables exhibiting no within-regime variation "
        r"were excluded from the interaction terms."
    )
)

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/2_simple_models/0_Dataset_2018_2025/Tables_Figures/chow_test_2018_2025.tex", "w") as f:
    f.write(results_tex)

# --- Visualization: DV with Structural Break Point ---
# NOTE: content only — no in-image title/note. Caption, label, and \fignote
# are set in the LaTeX \figure environment.
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["Global_TVL_USD_log_return_w"], color='black', linewidth=0.8)
ax.axvline(pd.to_datetime(SPLIT_DATE), color='red', linestyle='--', linewidth=1, label=f"Split: {SPLIT_DATE}")

ax.set_ylabel("Log Return", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.tick_params(axis='both', labelsize=8)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(
    "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/2_simple_models/0_Dataset_2018_2025/Tables_Figures/structural_break_tvl_2018_2025.png",
    dpi=300, bbox_inches='tight'
)
plt.close()

print(results_df)