import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import seaborn as sns
import matplotlib as mpl
import itertools


def displaySTFT(X, name=None):
    # display a modulation spectrogram, of shape (w1,w2,f,t)
    plt.figure(1)
    plt.pcolormesh(
        np.flipud(abs(np.squeeze(X))),
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',
    )
    plt.xticks([])
    plt.xlabel('')
    plt.yticks([])
    plt.ylabel('')
    if name is None:
        plt.show()
    else:
        plt.savefig(name)


def displayMSTFT(Z, name=None):
    # display a modulation spectrogram, of shape (w1,w2,f,t)
    plt.figure(1)
    (nF, nT) = Z.shape[2:4]
    for (f, t) in itertools.product(range(nF), range(nT)):
        plt.subplot(nF, nT, (nF-f-1) * nT+t+1)
        plt.pcolormesh(
            np.flipud(abs(Z[..., f, t])) ** 0.3,
            vmin=0,
            vmax=10,
            cmap='cubehelix_r',
        )
        plt.xticks([])
        plt.xlabel('')
        plt.yticks([])
        plt.ylabel('')

    f = plt.gcf()
    f.subplots_adjust(wspace=0, hspace=0)

    if name is None:
        plt.show()
    else:
        plt.savefig(name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Perform pitch estimation for given sensor data file.')

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
              'figure.figsize': fig_size * 1.6}

    plt.rcParams.update(params)

    cft = np.load('data/cft.npy')
    stft = np.load('data/gridstft.npy')

    print stft.shape
    print cft.shape

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].pcolormesh(
        np.flipud(abs(stft[..., 3, 0])) ** 0.5,
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',

    )
    ax[0, 0].set_aspect('equal')

    ax[0, 1].pcolormesh(
        np.flipud(abs(stft[..., 3, 4])) ** 0.5,
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',

    )
    ax[0, 1].set_aspect('equal')

    ax[0, 2].pcolormesh(
        np.flipud(abs(stft[..., 3, 2])) ** 0.5,
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',

    )
    ax[0, 2].set_aspect('equal')

    ax[1, 0].pcolormesh(
        np.flipud(abs(cft[..., 3, 0])) ** 0.3,
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',

    )
    ax[1, 0].set_aspect('equal')

    ax[1, 1].pcolormesh(
        np.flipud(abs(cft[..., 3, 4])) ** 0.3,
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',

    )
    ax[1, 1].set_aspect('equal')

    ax[1, 2].pcolormesh(
        np.flipud(abs(cft[..., 3, 2])) ** 0.3,
        vmin=0,
        vmax=10,
        cmap='cubehelix_r',

    )
    ax[1, 2].set_aspect('equal')

    for axis in ax:
        axis[0].set_xlim([0, 48])
        axis[0].set_ylim([0, 32])
        axis[1].set_xlim([0, 48])
        axis[1].set_ylim([0, 32])
        axis[2].set_xlim([0, 48])
        axis[2].set_ylim([0, 32])

    ax[0, 0].set_ylabel(r'$|STFT|$')
    ax[1, 0].set_ylabel(r'$|CFT|$')

    ax[1, 0].set_xlabel(r'Source A')
    ax[1, 1].set_xlabel(r'Source B')
    ax[1, 2].set_xlabel(r'Source A+B')

    # from matplotlib.patches import ConnectionPatch
    # xy = (31, 31)
    # con = ConnectionPatch(
    #     xyA=xy, xyB=xy, coordsA="data", coordsB="data",
    #     axesA=ax[0, 0], axesB=ax[1, 0]
    # )
    # ax[0, 0].add_artist(con)
    fig.tight_layout(w_pad=0.5, h_pad=0)
    fig.savefig(args.output, bbox_inches='tight', dpi=300)
