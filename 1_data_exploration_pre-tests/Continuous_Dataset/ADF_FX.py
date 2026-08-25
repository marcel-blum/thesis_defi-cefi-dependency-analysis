import pandas as pd
from statsmodels.tsa.stattools import adfuller

# load helper functions
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv", parse_dates=["date"])

## Foreign Exchange Rate Pairs ##
fx_cols = ["EUR_CNY_log_return" ,"EUR_USD_log_return", "USD_CNY_log_return"]

fx_results = []

for col in fx_cols:
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

    fx_results.append({
        "Variable": col,
        "ADF Statistic": f"{adf_result[0]:.4f}{stars}",
        "p-value": "$< 0.0001$" if p_value < 0.0001 else f"{p_value:.4f}",
        "Lags used": adf_result[2]
    })

fx_results_df = pd.DataFrame(fx_results)

print(fx_results_df)

fx_results_form = to_apa_latex_table(
    fx_results_df,
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/adf_fx.tex", "w") as f:
    f.write(fx_results_form)

plot_time_series(
    df=data,
    date_col="date",
    variables=["EUR_CNY_log_return", "EUR_USD_log_return", "USD_CNY_log_return"],
    titles=["EUR/CNY", "EUR/USD", "USD/CNY"],
    ylabel="Log Return",
    output_path="/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/fx_log_returns.png"
)