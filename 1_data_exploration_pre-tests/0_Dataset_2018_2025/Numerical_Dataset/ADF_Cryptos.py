import pandas as pd
from statsmodels.tsa.stattools import adfuller

# load helper functions
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv", parse_dates=["date"])

## Cryptocurrency Variables ##
crypto_cols = ["BTCUSDT_Log_Return", "ETHUSDT_Log_Return", "BNBUSDT_Log_Return", "SOLUSDT_Log_Return", "XRPUSDT_Log_Return", "BTCUSDT_Vol_LogGrowth", "ETHUSDT_Vol_LogGrowth", "BNBUSDT_Vol_LogGrowth", "SOLUSDT_Vol_LogGrowth", "XRPUSDT_Vol_LogGrowth"]

crypto_results = []

for col in crypto_cols:
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

    crypto_results.append({
        "Variable": col,
        "ADF Statistic": f"{round(adf_result[0], 4)}{stars}",
        "p-value": "$< 0.0001$" if p_value < 0.0001 else round(p_value, 4),
        "Lags used": adf_result[2]
    })

crypto_results_df = pd.DataFrame(crypto_results)

print(crypto_results_df)

crypto_results_form = to_apa_latex_table(
    crypto_results_df,
    "Augmented Dickey-Fuller Test: Cryptocurrency Close Prices and Trading Volume",
    "tab:adfcrypto",
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/adf_crypto.tex", "w") as f:
    f.write(crypto_results_form)

plot_time_series(
    df = data,
    date_col = "date",
    variables = ["BTCUSDT_Log_Return", "ETHUSDT_Log_Return", "BNBUSDT_Log_Return", "SOLUSDT_Log_Return", "XRPUSDT_Log_Return"],
    titles = ["BTC/USDT Return", "ETH/USDT Return", "BNB/USDT Return", "SOL/USDT Return", "XRP/USDT Return"],
    ylabel = "Log Return",
    figure_title = "Log Returns of Cryptocurrency Prices",
    figure_note = "Daily log returns retrieved via Binance API.",
    output_path = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/crypto_prices_log_returns.png"
)

plot_time_series(
    df = data,
    date_col = "date",
    variables = ["BTCUSDT_Vol_LogGrowth", "ETHUSDT_Vol_LogGrowth", "BNBUSDT_Vol_LogGrowth", "SOLUSDT_Vol_LogGrowth", "XRPUSDT_Vol_LogGrowth"],
    titles = ["BTC/USDT Volume", "ETH/USDT Volume", "BNB/USDT Volume", "SOL/USDT Volume", "XRP/USDT Volume"],
    ylabel = "Log Growth Rate",
    figure_title = "Log Growth Rates of Cryptocurrency Prices",
    figure_note = "Daily log growth rates retrieved via Binance API.",
    output_path = "/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/crypto_volume_log_growth.png"
)