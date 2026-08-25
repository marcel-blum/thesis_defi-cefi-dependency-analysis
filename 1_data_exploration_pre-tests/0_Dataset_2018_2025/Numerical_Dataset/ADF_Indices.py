import pandas as pd
from statsmodels.tsa.stattools import adfuller

# load helper functions
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Raw_Data/Dataset_1_Numerical_Before_Correlation_VIF.csv", parse_dates=["date"])

## Index & Commodity Variables ##
index_cols = [
    "CSI300_log_return", "GlobalGold_log_return", "MSCIEM_log_return",
    "MSCIWorld_log_return", "SP500_log_return", "SOX_log_return", "STOXX600_log_return"
]

index_results = []

for col in index_cols:
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

    index_results.append({
        "Variable": col,
        "ADF Statistic": f"{round(adf_result[0], 4)}{stars}",
        "p-value": "$< 0.0001$" if p_value < 0.0001 else round(p_value, 4),
        "Lags used": adf_result[2]
    })

index_results_df = pd.DataFrame(index_results)

print(index_results_df)

index_results_form = to_apa_latex_table(
    index_results_df,
    "Augmented Dickey-Fuller Test: Index and Commodity Log Returns",
    "tab:adfindices",
    note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
)
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/adf_indices.tex", "w") as f:
    f.write(index_results_form)

plot_time_series(
    df=data,
    date_col="date",
    variables=["SP500_log_return", "STOXX600_log_return", "CSI300_log_return"],
    titles=["S&P 500", "STOXX Europe 600", "CSI 300"],
    ylabel="Log Return",
    figure_title="Log Returns of Equity 07_Indices and Commodity (I): S&P 500, STOXX Europe 600, CSI 300",
    figure_note="Daily log returns retrieved via Refinitiv API.",
    output_path="/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/indices_log_returns_1.png"
)

plot_time_series(
    df=data,
    date_col="date",
    variables=["MSCIWorld_log_return", "MSCIEM_log_return", "SOX_log_return", "GlobalGold_log_return"],
    titles=["MSCI World", "MSCI Emerging Markets", "SOX", "Gold"],
    ylabel="Log Return",
    figure_title="Log Returns of Equity 07_Indices and Commodity (II): MSCI World, MSCI Emerging Markets, SOX, Gold",
    figure_note="Daily log returns retrieved via Refinitiv API.",
    output_path="/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/indices_log_returns_2.png"
)