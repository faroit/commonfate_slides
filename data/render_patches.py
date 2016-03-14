import commonfate
import pylab as plt
import numpy as np
np.random.seed(80)
import itertools
import soundfile as sf
import argparse
import yaml
import seaborn as sns
import matplotlib as mpl
import scipy.signal


def displaySTFT(X, name=None, limit=None):
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
        plt.axis((0, 48, 4, 28))
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
        displaySTFT(At, 'At.png')
        displaySTFT(xstft, 'stft.png', (0, 279, 0, 352))
        displaySTFT(stf_k[0], 'stft_k0.png', (0, 279, 0, 352))
        displaySTFT(stf_k[1], 'stft_k1.png', (0, 279, 0, 352))
        displayMSTFT(z[..., 0], 'cft.png')
        displayMSTFT(z_k[0][..., 0], 'cft_k0.png')
        displayMSTFT(z_k[1][..., 0], 'cft_k1.png')
        displayMSTFT(P, 'P.png')

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
