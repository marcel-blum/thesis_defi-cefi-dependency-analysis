import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
    exec(f.read())

data = pd.read_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Winsorized_Before_Selection.csv")

## VIF computation ##
# --- before selection ---
x_before = data.drop(columns=["date", "Global_TVL_USD_log_return_w"])
x_corr_before = x_before.corr().round(3)

vif_before = []
for index, name in enumerate(x_before.columns):
    vif_before.append({"Variable": name, "VIF": variance_inflation_factor(x_before.values, index)})

vif_before_df = pd.DataFrame(vif_before).sort_values(by=["VIF"], ascending=False).round(2)

# heatmap before selection (appendix / optional)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(x_corr_before, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, annot=False,
            cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax)
ax.set_title("Correlation Matrix of Independent Variables (Before Selection)",
              fontsize=12, style='italic', pad=20)
plt.tight_layout()
plt.savefig("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/heatmap_wins_before_2018_2025.png", dpi=300, bbox_inches='tight')
plt.close()

# --- after selection ---
#x_after = data.drop(columns=["date", "Global_TVL_USD_log_return", "MSCIWorld_log_return", "CHN_D_GDP"])
x_after = data.drop(columns=["date", "Global_TVL_USD_log_return_w", "MSCIWorld_log_return"])
x_corr_after = x_after.corr().round(3)

vif_after = []
for index, name in enumerate(x_after.columns):
    vif_after.append({"Variable": name, "VIF": variance_inflation_factor(x_after.values, index)})

vif_after_df = pd.DataFrame(vif_after).sort_values(by=["VIF"], ascending=False).round(2)

# heatmap after selection (main body)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(x_corr_after, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, annot=False,
            cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax)
ax.set_title("Correlation Matrix of Independent Variables (After Selection)",
              fontsize=12, style='italic', pad=20)
plt.tight_layout()
plt.savefig("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/heatmap_wins_after_2018_2025.png", dpi=300, bbox_inches='tight')
plt.close()

## Combined VIF table (before vs. after) ##
vif_combined_df = vif_before_df.rename(columns={"VIF": "VIF (Before)"}).merge(
    vif_after_df.rename(columns={"VIF": "VIF (After)"}), on="Variable", how="left"
).sort_values(by=["VIF (Before)"], ascending=False).reset_index(drop=True)

vif_combined_df["VIF (After)"] = vif_combined_df["VIF (After)"].fillna("—")

vif_combined_form = to_apa_latex_table(
    vif_combined_df,
    "Variance Inflation Factor Before and After Variable Selection",
    "tab:vif_combined"
)
with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/vif_combined_2018_2025.tex", "w") as f:
    f.write(vif_combined_form)

print(vif_combined_df)

# final dataset, as correlation investigation does not lead to variable drop
data.drop(columns = "MSCIWorld_log_return").to_csv("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Wins_Num_2018_2025_After_VIF_Selection.csv", index=False)