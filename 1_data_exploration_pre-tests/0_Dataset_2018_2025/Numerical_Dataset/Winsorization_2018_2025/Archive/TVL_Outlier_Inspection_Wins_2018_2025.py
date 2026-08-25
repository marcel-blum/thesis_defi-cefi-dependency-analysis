import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("/0_data/0_Dataset_2018_2025/Aggregated_Dataset_Num_2018_2025_Before_Selection.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
col = "Global_TVL_USD_log_return"

full  = df[col]
#early = df.loc[:"2019-01-31", col]   # pre-2019 (n=272)
#ref   = df.loc["2019-02-01":, col]   # post-2019 (n=2516)
early = df.loc[:"2020-08-11", col]   # pre-2020
ref   = df.loc["2020-08-12":, col]   # post-2020

# ------------------------------------------------------------------
# 2. Sensitivity table: percentile thresholds vs. clipping impact
# ------------------------------------------------------------------
percentiles = [0.001, 0.0025, 0.005, 0.01]
rows = []
for p in percentiles:
    lo, hi = full.quantile(p), full.quantile(1 - p)
    n_early = int(((early < lo) | (early > hi)).sum())
    n_ref   = int(((ref < lo) | (ref > hi)).sum())
    n_total = int(((full < lo) | (full > hi)).sum())
    rows.append({
        "Percentile":          f"{p*100:.2f}\\% / {100-p*100:.2f}\\%",
        "Lower Bound":         f"{lo:.4f}",
        "Upper Bound":         f"{hi:.4f}",
        #"Clipped (Pre-2019)":  f"{n_early} ({100*n_early/len(early):.1f}\\%)",
        #"Clipped (Post-2019)": f"{n_ref} ({100*n_ref/len(ref):.2f}\\%)",
        "Clipped (Pre-2020)": f"{n_early} ({100 * n_early / len(early):.1f}\\%)",
        "Clipped (Post-2020)": f"{n_ref} ({100 * n_ref / len(ref):.2f}\\%)",
        "Clipped (Total)":     str(n_total),
    })

winsor_diag = pd.DataFrame(rows)

# ------------------------------------------------------------------
# 3. Apply winsorizing at chosen percentile (full-sample based)
# ------------------------------------------------------------------
p_chosen = 0.0025   # 0.25% / 99.75% -> adjust if needed
lo, hi = full.quantile(p_chosen), full.quantile(1 - p_chosen)
df[f"{col}_w"] = full.clip(lo, hi)

# ------------------------------------------------------------------
# 4. LaTeX table via table_format.py
# ------------------------------------------------------------------
with open("/0_a_helpers/table_format.py") as f:
    exec(f.read())

#winsor_table = to_apa_latex_table(
#    winsor_diag,
#    "Sensitivity of Winsorizing Thresholds for Global TVL Log Returns (Full-Sample Percentiles)",
#    "tab:winsortvl20182025",
#    note=(r"\textit{Note}: Bounds computed as percentiles of the full sample "
#          r"(2018-05 to 2025-12, $n=2{,}788$). Pre-2019 refers to 2018-05 to 2019-01 "
#          r"($n=272$); Post-2019 refers to 2019-02 onwards ($n=2{,}516$).")
#)
winsor_table = to_apa_latex_table(
    winsor_diag,
    "Sensitivity of Winsorizing Thresholds for Global TVL Log Returns (Full-Sample Percentiles)",
    "tab:winsortvl20182025",
    note=(r"\textit{Note}: Bounds computed as percentiles of the full sample "
          r"2018-05-05 to 2025-12-21. Pre-2020 refers to 2018-05-05 to 2020-08-11 "
          r"Post-2020 refers to 2020-08-12 onwards.")
)

#with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/winsor_sensitivity_tvl_2018_2025.tex", "w") as f:
#    f.write(winsor_table)
with open(
        "/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/winsor_sensitivity_tvl_prepost2020_2018_2025.tex", "w") as f:
    f.write(winsor_table)

# ------------------------------------------------------------------
# 5. Descriptive statistics: Pre-2019 vs. Post-2019 vs. Full Sample
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 5. Descriptive statistics: Pre-2020 vs. Post-2020 vs. Full Sample
# ------------------------------------------------------------------
def extreme_share(series, thresh):
    n = (series.abs() > thresh).sum()
    return f"{n} ({100*n/len(series):.1f}\\%)"

desc_rows = []
#for label, s in [("Pre-2019 (2018-05 to 2019-01)",  early),
#                  ("Post-2019 (2019-02 to 2025-12)", ref),
#                  ("Full Sample (2018-05 to 2025-12)", full)]:
for label, s in [("Pre-2020 (2018-05-05 to 2020-08-11)",  early),
                  ("Post-2020 (2020-08-12 to 2025-12-21)", ref),
                  ("Full Sample (2018-05-05 to 2025-12-21)", full)]:
    desc_rows.append({
        "Period":     label,
        "N":          str(len(s)),
        "SD":         f"{s.std():.4f}",
        "IQR":        f"{s.quantile(0.75) - s.quantile(0.25):.4f}",
        "Min":        f"{s.min():.4f}",
        "Max":        f"{s.max():.4f}",
        "$|x|>0.5$":  extreme_share(s, 0.5),
        "$|x|>1.0$":  extreme_share(s, 1.0),
    })

desc_table = pd.DataFrame(desc_rows)

desc_latex = to_apa_latex_table(
    desc_table,
    "Descriptive Statistics of Global TVL Log Returns: Pre- vs. Post-2020",
    "tab:tvldescstats20182025",
    note=(r"\textit{Note}: IQR = interquartile range ($Q_{75}-Q_{25}$). "
          r"Pre-2020 refers to 2018-05-05 to 2020-08-11; "
          r"Post-2020 refers to 2020-08-12 onwards.")
)

#with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/tvl_descstats_pre_post2019_20182025.tex", "w") as f:
#    f.write(desc_latex)
with open("/1_data_exploration_pre-tests/0_Dataset_2018_2025/Tables_Figures/tvl_descstats_pre_post2020_20182025.tex", "w") as f:
    f.write(desc_latex)