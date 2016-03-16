import pylab as plt
import numpy as np
import itertools
import soundfile as sf
import argparse
import yaml
import commonfate
import matplotlib as mpl
import seaborn as sns
import math
np.random.seed(80)


def displaySTFT(X, name=None, limit=None):
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
    fig, ax = plt.subplots(1, 1)
    plt.figure(1)
    plt.pcolormesh(
        abs(np.squeeze(X)),
        vmin=0,
        vmax=20,
        cmap='cubehelix_r',
    )
    if limit is not None:
        plt.axis(limit)

    if name is None:
        plt.show()
    else:
        plt.savefig(name, bbox_inches='tight', dpi=300)


def displayMSTFT(Z, name=None, tricks=False):
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
    # display a modulation spectrogram, of shape (w1,w2,f,t)
    plt.figure(1)
    (nF, nT) = Z.shape[2:4]
    f, ax = plt.subplots(nF, nT)
    for (f, t) in itertools.product(range(nF), range(nT)):
        # plt.subplot(nF, nT, (nF-f-1) * nT+t+1)
        ax[f, t].pcolormesh(
            np.flipud(abs(Z[..., nF - f - 1, t])) ** 0.3,
            vmin=0,
            vmax=10,
            cmap='cubehelix_r',
        )

        ax[f, t].set_xticks([])
        ax[f, t].set_xlabel('')
        ax[f, t].set_yticks([])
        ax[f, t].set_ylabel('')

        if tricks:
            ax[f, t].axis((0, 48, 4, 28))

            if f == 0 and t == nT - 1:
                ax[f, t].yaxis.tick_right()
                ax[f, t].yaxis.set_label_position("right")
                ax[f, t].set_yticks([0, 32])
                ax[f, t].set_ylabel('')
                ax[f, t].xaxis.tick_top()
                ax[f, t].xaxis.set_label_position("top")
                ax[f, t].set_xticks([0, 48])
                ax[f, t].set_xlabel('')

            if f == nF - 1:
                ax[f, t].set_xticks([])
                ax[f, t].set_xlabel(str(t))

            if t == 0:
                ax[f, t].set_yticks([])
                ax[f, t].set_ylabel(
                    str(nF - f - 1),
                    rotation=0,
                    verticalalignment='center',
                    horizontalalignment='right'
                )

    f = plt.gcf()
    f.subplots_adjust(wspace=0, hspace=0)

    if name is None:
        plt.show()
    else:
        plt.savefig(name, bbox_inches='tight', dpi=300)


def process(
        signal,
        rate,
        pref,
        verbose=False,
        cluster=None,
        display=True,
        save=True
):

    W = (pref['W_A'], pref['W_B'])
    mhop = (pref['mhop_A'], pref['mhop_B'])

    print 'computing STFT'
    xstft = commonfate.transform.forward(xwave, pref['nfft'], pref['thop'])
    print xstft.shape

    xstft = xstft[:384-32, :]
    print xstft.shape

    # compute modulation STFT
    print 'computing modulation STFT'
    x = commonfate.transform.forward(xstft, W, mhop, real=False)

    print 'getting modulation spectrogram, shape:', x.shape
    z = np.abs(x) ** pref['alpha']

    # initialiase and fit the common fate model
    cfm = commonfate.model.CFM(z, nb_iter=100, nb_components=pref['J']).fit()

    # get the fitted factors
    (P, At, Ac) = cfm.factors

    # returns the of z approximation using the fitted factors
    z_hat = cfm.approx

    # source estimates
    estimates_k = []
    z_k = []
    stf_k = []

    for j in range(pref['J']):

        Fj = commonfate.model.hat(
            P[..., j][..., None],
            At[..., j][..., None],
            Ac[..., j][..., None],
        )

        yj = Fj / z_hat * x
        z_k.append(yj)
        # first compute back STFT
        yjstft = commonfate.transform.inverse(
            yj, fdim=2, hop=mhop, shape=xstft.shape, real=False
        )
        stf_k.append(yjstft)
        # then waveform
        wavej = commonfate.transform.inverse(
            yjstft, fdim=1, hop=pref['thop'], shape=signal.shape
        )
        estimates_k.append(wavej)
    #
    if display:
        displaySTFT(At, 'images/At.png')
        displaySTFT(xstft, 'images/stft.png', (0, 279, 0, 352))
        displaySTFT(stf_k[0], 'images/stft_k0.png', (0, 279, 0, 352))
        displaySTFT(stf_k[1], 'images/stft_k1.png', (0, 279, 0, 352))
        displayMSTFT(z[..., 0], 'images/cft.png', tricks=True)
        displayMSTFT(z_k[0][..., 0], 'images/cft_k0.png', tricks=True)
        displayMSTFT(z_k[1][..., 0], 'images/cft_k1.png', tricks=True)
        displayMSTFT(P, 'images/P.png', tricks=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Source Separation based on Common Fate Model')

    parser.add_argument('input', type=str, help='Input Audio File')

    args = parser.parse_args()

    # Parsing settings
    with open("data/settings.yml", 'r') as f:
        doc = yaml.load(f)

    pref = doc['general']

    filename = args.input

    # loading signal
    (xwave, fs) = sf.read(filename, always_2d=True)
    # xwave = scipy.signal.decimate(xwave, 2)
    # fs = fs/2
    out = process(xwave, fs, pref)
