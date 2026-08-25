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

raw = [
    ["MLR",         "0.025236", "0.018548", "0.0325"],
    ["Elastic Net", "0.024688", "0.018071", "0.0790"],
    ["GAM",         "0.025222", "0.018341", "0.0387"],
    ["XGBoost",     "0.024190", "0.017492", "0.1158"],
    ["SVR (RBF)",   "0.024309", "0.017763", "0.1071"],
    ["ARX",         "0.024536", "0.017873", "0.0903"],
    ["TFT",         "0.010807", "0.007093", "0.8235"],
]
data = [[m, fmt4(float(r)), fmt4(float(a)), fmt4(float(o))] for m, r, a, o in raw]
results_df = pd.DataFrame(data, columns=["Model", "RMSE", "MAE", "OOS R2"])

results_tex = to_apa_latex_table(
    results_df,
    note=r"\textit{Note}: RMSE and MAE in log-return units. OOS R$^2$ = $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$, computed on the held-out test set (last 20\% of observations, $N=392$) via rolling-origin one-step-ahead forecasts. All hyperparameters selected via 5-fold blocked cross-validation on the training set.",
    col_rename={"OOS R2": "OOS $R^2$"}
)

with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/9_Model_Comparison/Comparison/Tables_Figures/model_comparison.tex", "w") as f:
    f.write(results_tex)