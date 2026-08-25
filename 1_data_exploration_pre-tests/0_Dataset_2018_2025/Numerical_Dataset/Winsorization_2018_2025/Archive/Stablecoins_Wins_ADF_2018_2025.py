import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Load data (already contains winsorized TVL)
# ------------------------------------------------------------------
df = pd.read_csv(
    "/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv",
    parse_dates=["date"]
)
df.set_index("date", inplace=True)

# ------------------------------------------------------------------
# 2. Generic winsorization function (full-sample percentile based)
# ------------------------------------------------------------------
def winsorize_series(series, p):
    lo, hi = series.quantile(p), series.quantile(1 - p)
    n_clipped = int(((series < lo) | (series > hi)).sum())
    return series.clip(lo, hi), (lo, hi), n_clipped

# ------------------------------------------------------------------
# 3. Apply to Stablecoin MCap (p established via sensitivity analysis)
# ------------------------------------------------------------------
stbl_col = "Global_Stablecoins_Mcap_USD_log_return"
p_stbl = 0.0007  # 0.07% / 99.93%

df[f"{stbl_col}_w"], (lo, hi), n_clipped = winsorize_series(df[stbl_col], p_stbl)
df.drop(columns=[stbl_col], inplace=True)
print(f"Stablecoin MCap: p={p_stbl}, bounds=({lo:.4f}, {hi:.4f}), clipped={n_clipped}")

# ------------------------------------------------------------------
# 3b. Reorder columns: TVL_w second, Stablecoin_w third (date = index/1st)
# ------------------------------------------------------------------
cols = df.columns.tolist()
cols.remove("Global_TVL_USD_log_return_w")
cols.remove(f"{stbl_col}_w")
df = df[["Global_TVL_USD_log_return_w", f"{stbl_col}_w"] + cols]

# ------------------------------------------------------------------
# 4. Save (overwrite — now contains winsorized TVL and Stablecoin MCap)
# ------------------------------------------------------------------
df.to_csv(
    "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv"
)

# ------------------------------------------------------------------
# 5. ADF-Test
# ------------------------------------------------------------------
import pandas as pd
from statsmodels.tsa.stattools import adfuller

with open("/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv", parse_dates=["date"])

onchain_cols = ["Global_Stablecoins_Mcap_USD_log_return_w"]

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
    "Augmented Dickey-Fuller Test: Log Return/Growth of Global Stablecoin Market Capitalization",
    "tab:adfstablecoinafterwins20182025",
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open(
        "/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/adf_onchain_stablecoin_after_wins_2018_2025.tex", "w") as f:
    f.write(onchain_results_form)

plot_time_series(
    df=data,
    date_col="date",
    variables=["Global_Stablecoins_Mcap_USD_log_return_w"],
    titles=["Global Stablecoins MCap"],
    ylabel="Log Return/Log Growth Rate",
    figure_title="Log Growth of Global Stablecoin Market Capitalization",
    figure_note="Daily log growth rates were retrieved via DefiLlama API.",
    output_path="/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/stablecoin_log_returns_after_wins_2018_2025.png"
)