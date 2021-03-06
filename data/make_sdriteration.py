import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from numpy import median

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform pitch estimation for given sensor data file.')
    parser.add_argument(dest='input', type=str, default=None)
    parser.add_argument(dest='output', type=str, default=None,
                        help='pdf')

    args = parser.parse_args()

    plt.rc('text', usetex=True)
    plt.rc('font', family='FiraSans')

    mpl.rcParams['text.latex.preamble'] = [
        r"\usepackage[sfdefault,scaled=.85]{FiraSans}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{textcomp}",
        r"\usepackage[varqu,varl]{zi4}",
        r"\usepackage{amsmath,amsthm}",
        r"\usepackage[cmintegrals]{newtxsf}"
    ]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'FiraSans'
    mpl.rcParams['text.latex.unicode'] = 'True'

    sns.set()
    sns.set_context("paper")
    sns.set_style(
        "white", {
            "font.family":
            "serif", 'font.serif':
            'ptmrr8re'
        }
    )

    fig_width_pt = 244.6937  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27               # Convert pt to inch
    golden_mean = (math.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean      # height in inches
    fig_size = np.array([fig_width, fig_height])

    params = {'backend': 'ps',
              'axes.labelsize': 11,
              'legend.fontsize': 11,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'font.family': 'sans-serif',
              'font.sans-serif': 'FiraSans',
              'font.size': 11,
              'figure.figsize': fig_size}

    plt.rcParams.update(params)

    df = pd.read_pickle(args.input)
    df = df.query('k <= 6')
    # convert number of components to int
    df.k = df.k.astype(int)
    df.method = df.method.str.upper()

    g = sns.factorplot(
        x="k", y='SDR', hue="method", data=df,
        palette=sns.color_palette('muted'),
        dodge=True, estimator=median, legend_out=False, size=3.25, aspect=2.2,
        markers=["^", "v", "o", "s", "x"], linestyles=["-", "-", "-", "-", "-"]
    )
    g.set_axis_labels("Number of components", "SDR")
    # sns.despine(left=True)

    plt.legend(
        loc='lower right',
        ncol=5
    )

    g.fig.set_tight_layout(True)
    g.fig.savefig(args.output, bbox_inches='tight', dpi=300)
