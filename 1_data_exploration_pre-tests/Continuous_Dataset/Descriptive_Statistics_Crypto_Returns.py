import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

dataset = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv")

variables = ["BTCUSDT_Log_Return", "ETHUSDT_Log_Return", "BNBUSDT_Log_Return", "SOLUSDT_Log_Return", "XRPUSDT_Log_Return"]
data = dataset[variables]

## Descriptive Statistics ##
desc_stats_raw = pd.DataFrame({
    "Mean": data.mean(),
    "Median": data.median(),
    "Std. Dev.": data.std(),
    "Min": data.min(),
    "Max": data.max(),
    "Skewness": data.skew(),
    "Kurtosis": data.kurtosis()
})

print(desc_stats_raw.round(4))

desc_stats = desc_stats_raw.apply(lambda col: col.map(fmt4))
desc_stats_table = desc_stats.reset_index().rename(columns={"index": "Variable"})

desc_stats_form = to_apa_latex_table(
    desc_stats_table,
    note=r"\textit{Note}: Values below $0.0001$ in absolute magnitude are denoted as $<$0.0001."
)

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/Descriptive_Statistics/desc_stat_crypto_returns.tex", "w") as f:
    f.write(desc_stats_form)

## Histograms of Crypto Log Returns (content-only; overall caption/note set in LaTeX via \fignote) ##
titles = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, col, title in zip(axes, variables, titles):
    ax.hist(data[col], bins=50, color='gray', edgecolor='black', density=True)
    ax.set_title(title, fontstyle='italic', fontsize=12, pad=4)
    ax.set_xlabel("Log Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/Descriptive_Statistics/hist_crypto_returns.png", dpi=300, bbox_inches='tight')
plt.close()