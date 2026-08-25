import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# ==================================================================
# Winsorization Pipeline: TVL & Global Stablecoins MCap (2018-2025)
# Replaces:
#   - TVL_Outlier_Inspection_Wins_2018_2025.py
#   - TVL_Wins_ADF_2018_2025.py
#   - Stablecoins_Outlier_Inspection_Wins_2018_2025.py
#   - Stablecoins_Wins_ADF_2018_2025.py
#   - Merge_Tables_Plots.py
#
# Outputs:
#   1. ADF table & plot of ALL on-chain vars (before wins) -> .tex / .png
#   2. Sensitivity tables (TVL & Stablecoins)              -> .tex
#   3. Descriptive statistics tables (TVL & Stbl)          -> .tex
#   4. Winsorized dataset (both series, written once)      -> .csv
#   5. Merged ADF table (TVL + Stablecoins, after)         -> .tex
#   6. Merged time series plot (after winsorization)       -> .png
# ==================================================================

# ------------------------------------------------------------------
# 0. Paths & helpers
# ------------------------------------------------------------------
BASE          = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia"
DATA_IN       = f"{BASE}/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Before_Selection.csv"
DATA_OUT      = f"{BASE}/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv"
TABFIG        = f"{BASE}/1_data_exploration_pre-tests/0_Dataset_2018_2025/Numerical_Dataset/Winsorization_2018_2025/Tables_Figures"
TABFIG_BEFORE = TABFIG   # all tables & figures are stored in the same directory

with open(f"{BASE}/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open(f"{BASE}/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

# ------------------------------------------------------------------
# 1. Load raw data & define sample split
# ------------------------------------------------------------------
df = pd.read_csv(DATA_IN, parse_dates=["date"])
df.set_index("date", inplace=True)

SPLIT_PRE  = "2020-08-11"   # last day of pre-2020 period
SPLIT_POST = "2020-08-12"   # first day of post-2020 period

# ------------------------------------------------------------------
# 2. Generic functions
# ------------------------------------------------------------------
def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

def fmt_pval(p):
    if p is None or np.isnan(p):
        return "---"
    if p < 0.0001:
        return r"$<$0.0001"
    return f"{p:.4f}"

def fmt_pct(x):
    """Percentage value (already multiplied by 100): 4-decimal, '< 0.0001' rule."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if 0 <= x < 0.0001:
        return r"$<$0.0001\%"
    return f"{x:.4f}\\%"

def sig_stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def winsorize_series(series, p):
    """Winsorize at the p-th / (1-p)-th full-sample percentile."""
    lo, hi = series.quantile(p), series.quantile(1 - p)
    n_clipped = int(((series < lo) | (series > hi)).sum())
    return series.clip(lo, hi), (lo, hi), n_clipped


def sensitivity_table(series, percentiles, split_pre, split_post):
    """Clipping impact per candidate percentile, split pre/post-2020."""
    early = series.loc[:split_pre]
    ref   = series.loc[split_post:]
    rows = []
    for p in percentiles:
        lo, hi = series.quantile(p), series.quantile(1 - p)
        n_early = int(((early < lo) | (early > hi)).sum())
        n_ref   = int(((ref < lo) | (ref > hi)).sum())
        n_total = int(((series < lo) | (series > hi)).sum())
        rows.append({
            "Percentile":          f"{fmt4(p*100)}\\% / {fmt4(100 - p*100)}\\%",
            "Lower Bound":         fmt4(lo),
            "Upper Bound":         fmt4(hi),
            "Clipped (Pre-2020)":  f"{n_early} ({fmt_pct(100*n_early/len(early))})",
            "Clipped (Post-2020)": f"{n_ref} ({fmt_pct(100*n_ref/len(ref))})",
            "Clipped (Total)":     str(n_total),
        })
    return pd.DataFrame(rows)


def extreme_share(series, thresh):
    n = (series.abs() > thresh).sum()
    return f"{n} ({fmt_pct(100*n/len(series))})"


def descstats_table(series, split_pre, split_post, thresholds):
    """Descriptive statistics: pre-2020 vs. post-2020 vs. full sample."""
    early = series.loc[:split_pre]
    ref   = series.loc[split_post:]
    rows = []
    # Short period labels (full date ranges made the table too wide);
    # date ranges are given in the table note instead.
    for label, s in [("Pre-2020",   early),
                     ("Post-2020",  ref),
                     ("Full Sample", series)]:
        row = {
            "Period": label,
            "N":      str(len(s)),
            "SD":     fmt4(s.std()),
            "IQR":    fmt4(s.quantile(0.75) - s.quantile(0.25)),
            "Min":    fmt4(s.min()),
            "Max":    fmt4(s.max()),
        }
        for t in thresholds:
            row[f"$|x|>{t}$"] = extreme_share(s, t)
        rows.append(row)
    return pd.DataFrame(rows)


def adf_row(series, name):
    """Single ADF test result row with significance stars."""
    adf_result = adfuller(series.dropna(), autolag="AIC")
    stat, p_value, lags = adf_result[0], adf_result[1], adf_result[2]
    return {
        "Variable":      name,
        "ADF Statistic": fmt4(stat) + sig_stars(p_value),
        "p-value":       fmt_pval(p_value),
        "Lags used":     str(lags),
    }

# ------------------------------------------------------------------
# 3. On-chain variables BEFORE winsorization: ADF table & plot
# ------------------------------------------------------------------
onchain_cols = [
    "Global_TVL_USD_log_return",
    "Global_Stablecoins_Mcap_USD_log_return",
    "Global_Merchandise_Vol_USD_log_return",
    "Global_Gross_Revenue_USD_log_return",
]
onchain_titles = [
    "TVL",
    "Global Stablecoins MCap",
    "Global Merchandise Volume",
    "Global Gross Revenue",
]

adf_before_df = pd.DataFrame([adf_row(df[c], c) for c in onchain_cols])
print(adf_before_df)

# NOTE: to_apa_latex_table() returns ONLY the tabular fragment + note.
# Caption and label are supplied in-text via the \thesistable{...}{...}{...}{...} macro.
adf_before_tex = to_apa_latex_table(
    adf_before_df,
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open(f"{TABFIG_BEFORE}/adf_onchain_before_wins_2018_2025.tex", "w") as f:
    f.write(adf_before_tex)

# NOTE: plot_time_series() draws content only (no in-image title/note).
# Caption, label, and \fignote are set in the LaTeX \figure environment.
plot_time_series(
    df=df.reset_index(),
    date_col="date",
    variables=onchain_cols,
    titles=onchain_titles,
    ylabel="Log Return/Log Growth Rate",
    output_path=f"{TABFIG_BEFORE}/onchain_log_returns_before_wins_2018_2025.png"
)

# ------------------------------------------------------------------
# 4. TVL: sensitivity analysis & descriptive statistics (raw series)
# ------------------------------------------------------------------
tvl_col = "Global_TVL_USD_log_return"
tvl_raw = df[tvl_col]

tvl_sens = sensitivity_table(tvl_raw, [0.001, 0.0025, 0.005, 0.01],
                             SPLIT_PRE, SPLIT_POST)
tvl_sens_tex = to_apa_latex_table(
    tvl_sens,
    note=(r"\textit{Note}: Bounds computed as percentiles of the full sample "
          r"2018-05-05 to 2025-12-21. Pre-2020 refers to 2018-05-05 to 2020-08-11; "
          r"Post-2020 refers to 2020-08-12 onwards.")
)
with open(f"{TABFIG}/winsor_sensitivity_tvl_prepost2020_2018_2025.tex", "w") as f:
    f.write(tvl_sens_tex)

tvl_desc = descstats_table(tvl_raw, SPLIT_PRE, SPLIT_POST, thresholds=[0.5, 1.0])
tvl_desc_tex = to_apa_latex_table(
    tvl_desc,
    note=(r"\textit{Note}: IQR = interquartile range ($Q_{75}-Q_{25}$). "
          r"Pre-2020 refers to 2018-05-05 to 2020-08-11; "
          r"Post-2020 refers to 2020-08-12 onwards.")
)
with open(f"{TABFIG}/tvl_descstats_pre_post2020_20182025.tex", "w") as f:
    f.write(tvl_desc_tex)

# ------------------------------------------------------------------
# 5. Stablecoins: sensitivity analysis & descriptive statistics (raw)
# ------------------------------------------------------------------
stbl_col = "Global_Stablecoins_Mcap_USD_log_return"
stbl_raw = df[stbl_col]

stbl_sens = sensitivity_table(stbl_raw, [0.0005, 0.0007, 0.001, 0.0025],
                              SPLIT_PRE, SPLIT_POST)
stbl_sens_tex = to_apa_latex_table(
    stbl_sens,
    note=(r"\textit{Note}: Bounds computed as percentiles of the full sample "
          r"2018-05-05 to 2025-12-21. Pre-2020 refers to 2018-05-05 to 2020-08-11; "
          r"Post-2020 refers to 2020-08-12 onwards.")
)
with open(f"{TABFIG}/winsor_sensitivity_stablecoin_2018_2025.tex", "w") as f:
    f.write(stbl_sens_tex)

stbl_desc = descstats_table(stbl_raw, SPLIT_PRE, SPLIT_POST, thresholds=[0.1, 0.3])
stbl_desc_tex = to_apa_latex_table(
    stbl_desc,
    note=(r"\textit{Note}: IQR = interquartile range ($Q_{75}-Q_{25}$). "
          r"Pre-2020 refers to 2018-05-05 to 2020-08-11; "
          r"Post-2020 refers to 2020-08-12 onwards.")
)
with open(f"{TABFIG}/stablecoin_descstats_pre_post2020_20182025.tex", "w") as f:
    f.write(stbl_desc_tex)

# ------------------------------------------------------------------
# 6. Apply winsorization (both series) & save dataset ONCE
# ------------------------------------------------------------------
P_TVL  = 0.0025   # 0.25% / 99.75% (established via sensitivity analysis)
P_STBL = 0.0007   # 0.07% / 99.93% (established via sensitivity analysis)

df[f"{tvl_col}_w"],  (tvl_lo, tvl_hi),   tvl_clipped  = winsorize_series(tvl_raw,  P_TVL)
df[f"{stbl_col}_w"], (stbl_lo, stbl_hi), stbl_clipped = winsorize_series(stbl_raw, P_STBL)
df.drop(columns=[tvl_col, stbl_col], inplace=True)

print(f"TVL:         p={P_TVL},  bounds=({tvl_lo:.4f}, {tvl_hi:.4f}),  clipped={tvl_clipped}")
print(f"Stablecoins: p={P_STBL}, bounds=({stbl_lo:.4f}, {stbl_hi:.4f}), clipped={stbl_clipped}")

# Reorder: winsorized TVL first, Stablecoins second (date = index)
cols = df.columns.tolist()
cols.remove(f"{tvl_col}_w")
cols.remove(f"{stbl_col}_w")
df = df[[f"{tvl_col}_w", f"{stbl_col}_w"] + cols]

df.to_csv(DATA_OUT)

# ------------------------------------------------------------------
# 7. Merged ADF table (after winsorization) — single source of truth
# ------------------------------------------------------------------
adf_df = pd.DataFrame([
    adf_row(df[f"{tvl_col}_w"],  f"{tvl_col}_w"),
    adf_row(df[f"{stbl_col}_w"], f"{stbl_col}_w"),
])
print(adf_df)

adf_tex = to_apa_latex_table(
    adf_df,
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open(f"{TABFIG}/adf_onchain_after_wins_2018_2025.tex", "w") as f:
    f.write(adf_tex)

# ------------------------------------------------------------------
# 8. Merged time series plot (after winsorization)
# ------------------------------------------------------------------
plot_df = df.reset_index()

plot_time_series(
    df=plot_df,
    date_col="date",
    variables=[f"{tvl_col}_w", f"{stbl_col}_w"],
    titles=["TVL", "Global Stablecoins MCap"],
    ylabel="Log Return/Log Growth Rate",
    output_path=f"{TABFIG}/onchain_log_returns_after_wins_2018_2025.png"
)
