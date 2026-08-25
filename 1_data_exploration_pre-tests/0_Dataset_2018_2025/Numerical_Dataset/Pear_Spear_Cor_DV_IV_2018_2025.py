import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load table format function
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Wins_Num_2018_2025_After_VIF_Selection.csv")

y = data["Global_TVL_USD_log_return_w"]
x = data.drop(columns = ["date", "Global_TVL_USD_log_return_w"])

## Pearson Correlation Coefficient ##
pear_corr_tvl = x.corrwith(y).sort_values(ascending = False, key = abs).round(3)

## Spearman Correlation Coefficient ##
spear_corr_tvl = x.corrwith(y, method="spearman").sort_values(ascending=False, key=abs).round(3)

## Pearson vs. Spearman Comparison ##
corr_comparison_table = pd.DataFrame({
    "Variable": pear_corr_tvl.index,
    "Pearson": pear_corr_tvl.values,
    "Spearman": spear_corr_tvl.reindex(pear_corr_tvl.index).values
})

corr_comparison_table_form = to_apa_latex_table(
    corr_comparison_table,
    "Pearson vs. Spearman Correlation of Independent Variables with TVL",
    "tab:pear_spear_cor_tvl_2018_2025"
)

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/pearson_spearman_corr_comparison_w_2018_2025.tex", "w") as f:
    f.write(corr_comparison_table_form)

print(corr_comparison_table)

data.to_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Final_Dataset_numTVL_2018_2025.csv", index = False)