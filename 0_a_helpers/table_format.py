import re


def to_apa_latex_table(df, note=None, col_rename=None):
    """
    APA/Harvard-style LaTeX table helper (fragment mode).

    Emits ONLY the tabular body plus an optional note — no \\begin{table},
    no \\caption, no \\label, no font size command. Those are set in the
    main LaTeX document via the \\thesistable macro, e.g.:

        \\thesistable{\\footnotesize}
          {Model Comparison: Out-of-Sample Forecast Performance}
          {tab:modelcomparison}
          {/path/to/model_comparison.tex}

    Parameters:
    - df: pandas DataFrame (first column treated as row labels, left-aligned)
    - note: optional note string placed below the table (raw LaTeX allowed)
    - col_rename: optional dict mapping original column names to display names
    """
    display_cols = col_rename if col_rename else {}
    col_format = "l" + "c" * (len(df.columns) - 1)

    def safe_replace(col):
        return re.sub(r'(?<!\\)_', r'\\_', str(col))

    latex = f"\\begin{{tabular}}{{{col_format}}}" + "\n"
    latex += r"  \toprule" + "\n"

    headers = " & ".join(
        [f"\\textbf{{{safe_replace(display_cols.get(col, col))}}}" for col in df.columns]
    )
    latex += f"  {headers} \\\\ \n"
    latex += r"  \midrule" + "\n"

    for idx, row in df.iterrows():
        row_values = []
        for i, val in enumerate(row):
            if i == 0:
                row_values.append(re.sub(r'(?<!\\)_', r'\\_', str(val)))
            else:
                row_values.append(str(val))
        latex += f"  {' & '.join(row_values)} \\\\ \n"

    latex += r"  \bottomrule" + "\n"
    latex += r"\end{tabular}"

    if note:
        latex += "\n"
        latex += r"\par\vspace{4pt}" + "\n"
        latex += r"\begin{minipage}{\linewidth}" + "\n"
        latex += f"  \\footnotesize {note}" + "\n"
        latex += r"\end{minipage}"

    return latex


## example usage ##

# load function
# with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/table_format.py") as f:
#     exec(f.read())

# call function (without col_rename — standard case)
# onchain_results_form = to_apa_latex_table(
#     onchain_results_df,
#     note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1"
# )
# with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/adf_onchain.tex", "w") as f:
#     f.write(onchain_results_form)

# call function (with col_rename — e.g. for MLR table)
# coef_tex = to_apa_latex_table(
#     coef_df,
#     note=r"\textit{Note}: *** p$<$0.01, ** p$<$0.05, * p$<$0.1",
#     col_rename={"Coef": "$\\hat{\\beta}$"}
# )

## corresponding LaTeX preamble macro ##
# \newcommand{\thesistable}[4]{%
#   \begin{table}[ht]
#     \centering
#     \caption{\textit{#2}.}
#     \label{#3}
#     {#1
#     \input{#4}}
#   \end{table}}