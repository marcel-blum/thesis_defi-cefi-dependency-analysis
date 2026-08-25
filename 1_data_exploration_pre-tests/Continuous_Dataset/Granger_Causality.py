import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
INPUT_PATH  = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv"
OUTPUT_PATH = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures"

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

TARGET   = "Global_TVL_USD_log_return"
FEATURES = [c for c in df.columns if c not in ["date", TARGET]]

df_model = df[["date", TARGET] + FEATURES].dropna().reset_index(drop=True)

# ─────────────────────────────────────────────
# VARIABLE CATEGORIES
# ─────────────────────────────────────────────
CATEGORIES = {
    "Panel A: On-Chain Variables": [
        "Global_Stablecoins_Mcap_USD_log_return",
        "Global_Merchandise_Vol_USD_log_return",
        "Global_Gross_Revenue_USD_log_return",
    ],
    "Panel B: Cryptocurrency Returns": [
        "BTCUSDT_Log_Return",
        "ETHUSDT_Log_Return",
        "BNBUSDT_Log_Return",
        "XRPUSDT_Log_Return",
        "SOLUSDT_Log_Return",
    ],
    "Panel C: Cryptocurrency Trading Volume": [
        "BTCUSDT_Vol_LogGrowth",
        "ETHUSDT_Vol_LogGrowth",
        "BNBUSDT_Vol_LogGrowth",
        "XRPUSDT_Vol_LogGrowth",
        "SOLUSDT_Vol_LogGrowth",
    ],
    "Panel D: Exchange Rates": [
        "EUR_USD_log_return",
        "EUR_CNY_log_return",
        "USD_CNY_log_return",
    ],
    "Panel E: Monetary Policy": [
        "USA_Rate_Hike",
        "USA_Rate_Cut",
        "EZ_Rate_Hike",
        "EZ_Rate_Cut",
        "China_Rate_Cut",
    ],
    "Panel F: Macroeconomic Variables": [
        "USA_D_GDP",
        "EZ_D_GDP",
        "USA_D_CPI",
        "EZ_D_CPI",
        "CHN_D_CPI",
        "USA_D_UR",
        "EZ_D_UR",
        "CHN_D_UR",
    ],
    "Panel G: Money Supply / Liquidity": [
        "USA_D_Liquidity",
        "EZ_D_Liquidity",
        "CHN_D_Liquidity",
    ],
    "Panel H: Equity \\& Commodity Indices": [
        "SP500_log_return",
        "STOXX600_log_return",
        "CSI300_log_return",
        "MSCIEM_log_return",
        "GlobalGold_log_return",
        "SOX_log_return",
    ],
}

# ─────────────────────────────────────────────
# GRANGER CAUSALITY SETTINGS
# ─────────────────────────────────────────────
MAX_LAG   = 5
ALPHA     = 0.05
N_TESTS   = len(FEATURES)
ALPHA_BON = ALPHA / N_TESTS

print(f"Running Granger Causality Tests: {N_TESTS} predictors → {TARGET}")
print(f"maxlag = {MAX_LAG} | α = {ALPHA} | α_Bonferroni = {ALPHA_BON:.6f}\n")

# ─────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────
results = {}

for feat in FEATURES:
    data = df_model[[TARGET, feat]].values

    try:
        gc_res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)

        lag_pvals = {}
        for lag in range(1, MAX_LAG + 1):
            f_stat = gc_res[lag][0]["ssr_ftest"][0]
            p_val  = gc_res[lag][0]["ssr_ftest"][1]
            lag_pvals[lag] = {"F": f_stat, "p": p_val}

        min_lag   = min(lag_pvals, key=lambda l: lag_pvals[l]["p"])
        min_pval  = lag_pvals[min_lag]["p"]
        min_fstat = lag_pvals[min_lag]["F"]

        results[feat] = {
            "best_lag": min_lag,
            "F":        min_fstat,
            "p":        min_pval,
            "sig_raw":  min_pval < ALPHA,
            "sig_bon":  min_pval < ALPHA_BON,
        }

        print(f"  {feat:<50} lag={min_lag}  F={min_fstat:.4f}  p={min_pval:.6f}  "
              f"{'*** Bonf.' if min_pval < ALPHA_BON else ('* raw' if min_pval < ALPHA else '')}")

    except Exception as e:
        print(f"  [ERROR] {feat}: {e}")
        results[feat] = {
            "best_lag": None, "F": None, "p": None,
            "sig_raw": False, "sig_bon": False,
        }

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def fmt_num(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if 0 <= x < 0.0001:
        return "$<$0.0001"
    return f"{x:.4f}"

def format_pval(p):
    return fmt_num(p)

def sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < ALPHA_BON:
        return "$^{***}$"
    if p < ALPHA:
        return "$^{*}$"
    return ""

def escape_var(name):
    return name.replace("_", "\\_")

# ─────────────────────────────────────────────
# BUILD GROUPED LATEX TABLE
# Sig. (Bonf.) column removed — stars in p-Value
# column are sufficient (standard in top journals).
# ─────────────────────────────────────────────
n_sig_raw = sum(1 for r in results.values() if r["sig_raw"])
n_sig_bon = sum(1 for r in results.values() if r["sig_bon"])

headers = ["Variable", "Lag", "$F$-Statistic", "$p$-Value"]

# NOTE: this builds ONLY the tabular fragment + note (no \begin{table}, \caption,
# \label) — consistent with to_apa_latex_table()'s fragment convention. Caption
# and label are supplied in-text via the \thesistable{...}{...}{...}{...} macro.
def build_latex_table():
    tex  = r"{\renewcommand{\arraystretch}{0.90}" + "\n"
    tex += r"\begin{tabular}{lccc}" + "\n"
    tex += r"    \toprule" + "\n"
    tex += "    " + " & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\ \n"
    tex += r"    \midrule" + "\n"

    first_panel = True
    for panel_name, var_list in CATEGORIES.items():
        if not first_panel:
            tex += r"    \midrule" + "\n"
        first_panel = False

        tex += f"    \\multicolumn{{4}}{{l}}{{\\textit{{{panel_name}}}}} \\\\ \n"
        tex += r"    \cmidrule(lr){1-4}" + "\n"

        for feat in var_list:
            if feat not in results:
                continue
            r = results[feat]
            lag_str = str(r["best_lag"]) if r["best_lag"] is not None else "---"
            f_str   = fmt_num(r["F"])
            p_str   = format_pval(r["p"]) + sig_stars(r["p"])

            tex += f"    {escape_var(feat)} & {lag_str} & {f_str} & {p_str} \\\\ \n"

    tex += r"    \bottomrule" + "\n"
    tex += r"  \end{tabular}}" + "\n"

    note = (
        r"\textit{Note}: H$_0$: Variable $X_i$ does not Granger-cause "
        r"\textit{Global\_TVL\_USD\_log\_return}. "
        r"F-test (SSR) reported at the lag $\ell \in \{1, \dots, 5\}$ yielding "
        r"the lowest p-value. "
        rf"Bonferroni-corrected significance threshold: "
        rf"$\alpha^* = 0.05 / {N_TESTS} \approx {ALPHA_BON:.4f}$. "
        rf"{n_sig_bon} of {N_TESTS} variables are significant after Bonferroni "
        rf"correction; {n_sig_raw} of {N_TESTS} at the uncorrected "
        r"$\alpha = 0.05$ level. "
        r"$^{***}$ p$<\alpha^*$ (Bonferroni); $^{*}$ p$<$0.05 (uncorrected)."
    )
    # NOTE: \par + \begin{minipage}{\linewidth} forces a full-width block below
    # the table (matches to_apa_latex_table()'s note convention). Without this,
    # the footnote stays in centering's restricted horizontal mode and can drift
    # to the right of the last table row instead of starting a new line beneath it.
    tex += r"\par\vspace{4pt}" + "\n"
    tex += r"\begin{minipage}{\linewidth}" + "\n"
    tex += f"  \\footnotesize {note}" + "\n"
    tex += r"\end{minipage}"
    return tex

granger_tex = build_latex_table()

with open(f"{OUTPUT_PATH}/granger_causality.tex", "w") as f:
    f.write(granger_tex)
print("\n✓ Granger causality table saved.")

# ─────────────────────────────────────────────
# SUMMARY PRINT
# ─────────────────────────────────────────────
print(f"\n─── SUMMARY ───")
print(f"Total tests           : {N_TESTS}")
print(f"Bonferroni α*         : {ALPHA_BON:.6f}")
print(f"Significant (raw)     : {n_sig_raw} / {N_TESTS}")
print(f"Significant (Bonf.)   : {n_sig_bon} / {N_TESTS}")
print(f"\nVariables significant after Bonferroni correction:")
for feat, r in sorted(results.items(), key=lambda x: x[1]["p"] or 1):
    if r["sig_bon"]:
        print(f"  {feat:<50} p={r['p']:.6f}  lag={r['best_lag']}")