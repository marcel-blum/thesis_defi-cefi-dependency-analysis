import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load table format function
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv")

y = data["Global_TVL_USD_log_return"]
x = data.drop(columns = ["date", "Global_TVL_USD_log_return"])

# ─────────────────────────────────────────────
# HELPER: 4-DECIMAL FORMATTING
# ─────────────────────────────────────────────
def fmt4(x):
    """Round to 4 decimals; magnitudes below 0.0001 shown as '< 0.0001' (sign-aware)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if 0 <= x < 0.0001:
        return r"$<$0.0001"
    if -0.0001 < x < 0:
        return r"$>$-0.0001"
    return f"{x:.4f}"

## Pearson Correlation Coefficient ##
pear_corr_tvl = x.corrwith(y).sort_values(ascending = False, key = abs)

## Spearman Correlation Coefficient ##
spear_corr_tvl = x.corrwith(y, method="spearman").sort_values(ascending=False, key=abs)

## Pearson vs. Spearman Comparison ##
corr_comparison_table = pd.DataFrame({
    "Variable": pear_corr_tvl.index,
    "Pearson": pear_corr_tvl.values,
    "Spearman": spear_corr_tvl.reindex(pear_corr_tvl.index).values
})

# display copy: 4-decimal / "< 0.0001" formatting (sign-aware)
corr_comparison_display = corr_comparison_table.copy()
corr_comparison_display["Pearson"]  = corr_comparison_display["Pearson"].apply(fmt4)
corr_comparison_display["Spearman"] = corr_comparison_display["Spearman"].apply(fmt4)

# NOTE: to_apa_latex_table() returns ONLY the tabular fragment + note.
# Caption and label are supplied in-text via the \thesistable{...}{...}{...}{...} macro.
corr_comparison_table_form = to_apa_latex_table(
    corr_comparison_display,
    note=(
        r"\textit{Note}: Pearson coefficients measure linear association; Spearman "
        r"coefficients measure monotonic (rank-based) association with "
        r"\textit{Global\_TVL\_USD\_log\_return}. Variables sorted by descending "
        r"absolute Pearson correlation."
    )
)

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/pearson_spearman_corr_comparison.tex", "w") as f:
    f.write(corr_comparison_table_form)

print(corr_comparison_table)
