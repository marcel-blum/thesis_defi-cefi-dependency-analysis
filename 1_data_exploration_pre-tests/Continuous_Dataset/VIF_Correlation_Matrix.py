import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Raw_Data/Dataset_1_Numerical_Before_Correlation_VIF.csv")

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

## VIF computation ##
# --- before selection ---
x_before = data.drop(columns=["date", "Global_TVL_USD_log_return"])
x_corr_before = x_before.corr()

vif_before = []
for index, name in enumerate(x_before.columns):
    vif_before.append({"Variable": name, "VIF": variance_inflation_factor(x_before.values, index)})

vif_before_df = pd.DataFrame(vif_before).sort_values(by=["VIF"], ascending=False)

# heatmap before selection (appendix / optional)
# NOTE: no in-image title — caption/label are set in the LaTeX \figure environment.
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(x_corr_before, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, annot=False,
            cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax)
plt.tight_layout()
plt.savefig("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/heatmap_before.png", dpi=300, bbox_inches='tight')
plt.close()

# --- after selection ---
x_after = data.drop(columns=["date", "Global_TVL_USD_log_return", "MSCIWorld_log_return", "CHN_D_GDP"])
x_corr_after = x_after.corr()

vif_after = []
for index, name in enumerate(x_after.columns):
    vif_after.append({"Variable": name, "VIF": variance_inflation_factor(x_after.values, index)})

vif_after_df = pd.DataFrame(vif_after).sort_values(by=["VIF"], ascending=False)

# heatmap after selection (main body)
# NOTE: no in-image title — caption/label are set in the LaTeX \figure environment.
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(x_corr_after, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, annot=False,
            cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax)
plt.tight_layout()
plt.savefig("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/heatmap_after.png", dpi=300, bbox_inches='tight')
plt.close()

## Combined VIF table (before vs. after) ##
vif_combined_df = vif_before_df.rename(columns={"VIF": "VIF (Before)"}).merge(
    vif_after_df.rename(columns={"VIF": "VIF (After)"}), on="Variable", how="left"
).sort_values(by=["VIF (Before)"], ascending=False).reset_index(drop=True)

# display copy: 4-decimal / "< 0.0001" formatting; "---" for variables dropped after selection
vif_combined_display = vif_combined_df.copy()
vif_combined_display["VIF (Before)"] = vif_combined_display["VIF (Before)"].apply(fmt4)
vif_combined_display["VIF (After)"]  = vif_combined_display["VIF (After)"].apply(fmt4)

# NOTE: to_apa_latex_table() returns ONLY the tabular fragment + note.
# Caption and label are supplied in-text via the \thesistable{...}{...}{...}{...} macro.
vif_combined_form = to_apa_latex_table(
    vif_combined_display,
    note=(
        r"\textit{Note}: Variance Inflation Factor (VIF) values above the "
        r"conventional threshold of 10 indicate problematic multicollinearity. "
        r"\textit{VIF (After)} reports recalculated values following the exclusion "
        r"of \textit{MSCIWorld\_log\_return} and \textit{CHN\_D\_GDP}; a dash (---) "
        r"denotes variables no longer present in the final specification."
    )
)
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/vif_combined.tex", "w") as f:
    f.write(vif_combined_form)

print(vif_combined_df)

# final dataset, as correlation investigation does not lead to variable drop
data.drop(columns=["MSCIWorld_log_return", "CHN_D_GDP"]).to_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/Final_Dataset_numTVL.csv", index=False)
