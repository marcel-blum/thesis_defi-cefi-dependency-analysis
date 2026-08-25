import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import os

def plot_time_series(
    df,
    date_col,
    variables,
    titles=None,
    ylabel="",
    output_path=None,
    figsize=(12, 4)
):
    """
    APA/Harvard-style time series plot helper (content-only mode).

    The PNG contains ONLY the plotted content. Figure title, note, and
    label are set in the main LaTeX document:

        \\begin{figure}[ht]
            \\centering
            \\includegraphics[width=\\textwidth]{/path/to/plot.png}
            \\fignote{Daily log returns retrieved via DefiLlama API.}
            \\caption{\\textit{Log Returns of Blockchain On-Chain Variables}.}
            \\label{fig:onchainlogreturns}
        \\end{figure}

    Panel titles (subplot titles) are rendered only for multi-panel
    figures, where they identify the individual panels. For single-panel
    figures the LaTeX caption carries the description, so no in-image
    title is drawn.

    Parameters:
    - df: pandas DataFrame
    - date_col: name of the date column
    - variables: list of column names to plot
    - titles: list of panel titles (one per variable); used only if
      more than one variable is plotted
    - ylabel: y-axis label
    - output_path: full path to save the figure (.png)
    - figsize: tuple for figure size per subplot (width, height)
    """

    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n))

    if n == 1:
        axes = [axes]
    if titles is None:
        titles = [""] * n

    for i, (col, title) in enumerate(zip(variables, titles)):
        ax = axes[i]
        ax.plot(df[date_col], df[col], color='black', linewidth=0.8)
        if n > 1 and title:  # panel titles only for multi-panel figures
            ax.set_title(title, fontstyle='italic', fontsize=12, pad=4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.tick_params(axis='both', labelsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


## example usage ##

# load function
# with open("/Users/marcel/PycharmProjects/Master_Thesis_Pavia/0_a_helpers/time_series_plot_figure_format.py") as f:
#     exec(f.read())

# call function
# plot_time_series(
#     df=data,
#     date_col="date",
#     variables=["Global_TVL_USD_log_return", "Global_Stablecoins_Mcap_USD_log_return", "Global_Merchandise_Vol_USD_log_return", "Global_Gross_Revenue_USD_log_return"],
#     titles=["TVL", "Global Stablecoins MCap", "GMV", "GGR"],
#     ylabel="Log Return/Log Growth Rates",
#     output_path="/Users/marcel/PycharmProjects/Master_Thesis_Pavia/1_data_exploration_pre-tests/Tables_Figures/onchain_log_returns.png"
# )

## corresponding LaTeX preamble macro ##
# \newcommand{\fignote}[1]{%
#   \par\vspace{4pt}%
#   \begin{minipage}{\textwidth}\footnotesize \textit{Note}. #1\end{minipage}}