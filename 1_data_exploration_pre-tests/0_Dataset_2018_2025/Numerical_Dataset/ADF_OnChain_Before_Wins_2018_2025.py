import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# load helper functions
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Before_Selection.csv", parse_dates=["date"])

# ─────────────────────────────────────────────
# HELPER: 4-DECIMAL FORMATTING
# ─────────────────────────────────────────────
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

def sig_stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

## Blockchain On-Chain Variables ##
onchain_cols = ["Global_TVL_USD_log_return", "Global_Stablecoins_Mcap_USD_log_return", "Global_Merchandise_Vol_USD_log_return", "Global_Gross_Revenue_USD_log_return"]

onchain_results = []

for col in onchain_cols:
    adf_result = adfuller(data[col].dropna(), autolag="AIC")
    stat, p_value, lags = adf_result[0], adf_result[1], adf_result[2]

    onchain_results.append({
        "Variable":      col,
        "ADF Statistic": fmt4(stat) + sig_stars(p_value),
        "p-value":       fmt_pval(p_value),
        "Lags used":     str(lags)
    })

onchain_results_df = pd.DataFrame(onchain_results)

print(onchain_results_df)

# NOTE: to_apa_latex_table() returns ONLY the tabular fragment + note.
# Caption and label are supplied in-text via the \thesistable{...}{...}{...}{...} macro.
onchain_results_form = to_apa_latex_table(
    onchain_results_df,
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/adf_onchain_before_wins_2018_2025.tex", "w") as f:
    f.write(onchain_results_form)

# NOTE: plot_time_series() draws content only (no in-image title/note).
# Caption, label, and \fignote are set in the LaTeX \figure environment.
plot_time_series(
    df=data,
    date_col="date",
    variables=["Global_TVL_USD_log_return", "Global_Stablecoins_Mcap_USD_log_return", "Global_Merchandise_Vol_USD_log_return", "Global_Gross_Revenue_USD_log_return"],
    titles=["TVL", "Global Stablecoins MCap", "GMV", "GGR"],
    ylabel="Log Return/Log Growth Rates",
    output_path="/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/onchain_log_returns_before_wins_2018_2025.png"
)
