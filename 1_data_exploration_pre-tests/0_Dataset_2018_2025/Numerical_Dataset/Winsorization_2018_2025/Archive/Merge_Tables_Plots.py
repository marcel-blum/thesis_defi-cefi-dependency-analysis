import pandas as pd
from statsmodels.tsa.stattools import adfuller

with open("/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv", parse_dates=["date"])

## Blockchain On-Chain Variables (winsorized) ##
onchain_cols   = ["Global_TVL_USD_log_return_w", "Global_Stablecoins_Mcap_USD_log_return_w"]
onchain_titles = ["TVL", "Global Stablecoins MCap"]

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
    "Augmented Dickey-Fuller Test: Log Return/Growth of Blockchain On-Chain Variables (Winsorized)",
    "tab:adftvlstablesafterwins20182025",
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open("/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/adf_onchain_after_wins_2018_2025.tex", "w") as f:
    f.write(onchain_results_form)

plot_time_series(
    df=data,
    date_col="date",
    variables=onchain_cols,
    titles=onchain_titles,
    ylabel="Log Return/Log Growth Rate",
    figure_title="Log Return/Growth of Blockchain On-Chain Variables (Winsorized)",
    figure_note="Daily log returns/growth rates were retrieved via DefiLlama API.",
    output_path="/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/onchain_log_returns_after_wins_2018_2025.png"
)