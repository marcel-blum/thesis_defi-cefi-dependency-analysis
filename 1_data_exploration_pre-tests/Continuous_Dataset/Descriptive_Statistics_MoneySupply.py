import pandas as pd

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

variables = ["USA_D_Liquidity", "EZ_D_Liquidity", "CHN_D_Liquidity"]

freq_table = pd.DataFrame({
    "Variable": variables,
    "N (0)": [(dataset[v] == 0).sum() for v in variables],
    "N (1)": [(dataset[v] == 1).sum() for v in variables],
    "Share": [fmt4((dataset[v] == 1).mean()) for v in variables]
})

print(freq_table)

freq_table_form = to_apa_latex_table(
    freq_table,
    note=(
        r"\textit{Note}: Share denotes the proportion of observations coded 1. "
        r"Values below $0.0001$ in absolute magnitude are denoted as $<$0.0001."
    )
)

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/Descriptive_Statistics/freq_money_supply.tex", "w") as f:
    f.write(freq_table_form)