import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(
    "/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Before_Selection.csv",
    parse_dates=["date"]
)
df.set_index("date", inplace=True)

# ------------------------------------------------------------------
# 2. Generic winsorization function (full-sample percentile based)
# ------------------------------------------------------------------
def winsorize_series(series, p):
    """
    Winsorizes a series at the p-th / (1-p)-th percentile of its own
    full-sample distribution.

    Parameters
    ----------
    series : pd.Series
    p : float
        Tail percentile (e.g., 0.0025 for 0.25% / 99.75%).

    Returns
    -------
    winsorized : pd.Series
    bounds : tuple (lower, upper)
    n_clipped : int
    """
    lo, hi = series.quantile(p), series.quantile(1 - p)
    n_clipped = int(((series < lo) | (series > hi)).sum())
    return series.clip(lo, hi), (lo, hi), n_clipped

# ------------------------------------------------------------------
# 3. Apply to TVL (p established via sensitivity analysis)
# ------------------------------------------------------------------
tvl_col = "Global_TVL_USD_log_return"
p_tvl = 0.0025  # 0.25% / 99.75%

df[f"{tvl_col}_w"], (lo, hi), n_clipped = winsorize_series(df[tvl_col], p_tvl)
df.drop(columns=[tvl_col], inplace=True)
print(f"TVL: p={p_tvl}, bounds=({lo:.4f}, {hi:.4f}), clipped={n_clipped}")

# ------------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------------
df.to_csv(
    "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv"
)

# ------------------------------------------------------------------
# 5. ADF-Test
# ------------------------------------------------------------------

import pandas as pd
from statsmodels.tsa.stattools import adfuller

# load helper functions
with open("/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv", parse_dates=["date"])

## Blockchain On-Chain Variables ##
onchain_cols = ["Global_TVL_USD_log_return_w"]

onchain_results = []

for col in onchain_cols:
    adf_result = adfuller(data[col].dropna(), autolag="AIC")
    p_value = adf_result[1]

    if p_value < 0.01:
        stars = "***"
    elif p_value < 0.05:
        stars = "**"
    elif p_value < 0.10:
        stars = "*"
    else:
        stars = ""

    onchain_results.append({
        "Variable": col,
        "ADF Statistic": f"{round(adf_result[0], 4)}{stars}",
        "p-value": "$< 0.0001$" if p_value < 0.0001 else round(p_value, 4),
        "Lags used": adf_result[2]
    })

onchain_results_df = pd.DataFrame(onchain_results)

print(onchain_results_df)

onchain_results_form = to_apa_latex_table(
    onchain_results_df,
    "Augmented Dickey-Fuller Test: Log Return/Growth of Total Value Locked",
    "tab:adftvlafterwins20182025",
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open("/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/adf_onchain_after_wins_2018_2025.tex", "w") as f:
    f.write(onchain_results_form)

plot_time_series(
    df=data,
    date_col="date",
    variables=["Global_TVL_USD_log_return_w"],
    titles=["TVL"],
    ylabel="Log Return",
    figure_title="Log Return of Total Value Locked",
    figure_note="Daily log returns were retrieved via DefiLlama API.",
    output_path="/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/tvl_log_returns_after_wins_2018_2025.png"
)