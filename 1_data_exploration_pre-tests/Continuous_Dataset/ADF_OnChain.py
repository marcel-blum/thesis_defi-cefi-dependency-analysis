import pandas as pd
from statsmodels.tsa.stattools import adfuller

# load helper functions
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv", parse_dates=["date"])

## Blockchain On-Chain Variables ##
onchain_cols = ["Global_TVL_USD_log_return", "Global_Stablecoins_Mcap_USD_log_return", "Global_Merchandise_Vol_USD_log_return", "Global_Gross_Revenue_USD_log_return"]

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
        "ADF Statistic": f"{adf_result[0]:.4f}{stars}",
        "p-value": "$< 0.0001$" if p_value < 0.0001 else f"{p_value:.4f}",
        "Lags used": adf_result[2]
    })

onchain_results_df = pd.DataFrame(onchain_results)

print(onchain_results_df)

onchain_results_form = to_apa_latex_table(
    onchain_results_df,
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/adf_onchain.tex", "w") as f:
    f.write(onchain_results_form)

plot_time_series(
    df=data,
    date_col="date",
    variables=["Global_TVL_USD_log_return", "Global_Stablecoins_Mcap_USD_log_return", "Global_Merchandise_Vol_USD_log_return", "Global_Gross_Revenue_USD_log_return"],
    titles=["TVL", "Global Stablecoins MCap", "GMV", "GGR"],
    ylabel="Log Return/Log Growth Rates",
    output_path="/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/onchain_log_returns.png"
)