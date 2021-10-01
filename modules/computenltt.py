import pickle as pk
import numpy as np
from scipy import signal
import os


def autocorr(x):
    result = signal.correlate(x, x, mode='full', method='fft')
    v = [result[i] / (len(x) - abs(i - (len(x)) + 1)) for i in range(len(result))]
    return np.array(v[int(result.size / 2):])


def computenltt(root, Np, L, nk, cp, deltat, tdump):
    if os.path.exists(root + 'chk.pkl'):
        with open(root + 'enk.pkl', 'rb') as g:
            enkb = pk.load(g)
            enk = np.transpose(np.array(enkb))
        with open(root + 'dipenkx.pkl', 'rb') as g:
            dipenkxb = pk.load(g)
            dipenkx = np.transpose(np.array(dipenkxb))
        with open(root + 'dipenky.pkl', 'rb') as g:
            dipenkyb = pk.load(g)
            dipenky = np.transpose(np.array(dipenkyb))
        with open(root + 'chk.pkl', 'rb') as g:
            chkb = pk.load(g)
            chk = np.transpose(np.array(chkb))
        with open(root + 'dipkx.pkl', 'rb') as g:
            dipkxb = pk.load(g)
            dipkx = np.transpose(np.array(dipkxb))
        with open(root + 'dipky.pkl', 'rb') as g:
            dipkyb = pk.load(g)
            dipky = np.transpose(np.array(dipkyb))
        nsnap = int(len(enkb)/3)
    else:
        raise ValueError

    # nsnap, enk, dipenkx, dipenky, chk, dipkx, dipky = computekft(root, filename, Np, L, posox, nk, ntry, natpermol)

    ndata = int(enk.shape[1])

    enkcorr = np.reshape(enk, (nk, int(ndata / 3), 3))

    nblocks = 10

    tblock = int(enkcorr.shape[1] / nblocks)

    tinblock = int(tblock / 2)

    rho = Np / (6.022e23 * L ** 3 * 1.e-30)  # mol/m^3

    fac = rho * cp  # J/k/m^3

    dt = deltat * tdump  # ps

    corr = np.zeros((nblocks, tinblock), dtype=np.complex_)

    chi = np.var(enk[:, :], axis=1)  # (Kcal*m)**2

    corren = np.zeros((nblocks, nk - 1, int(tinblock / 2) + 1), dtype=np.complex_)

    ft = np.zeros((nk - 1, int(tinblock / 2) + 1), dtype=np.complex_)

    corrk = np.zeros((nblocks, nk-1, tinblock), dtype=np.complex_)

    for t in range(nblocks):

        print(t)

        for j in range(1, nk):
            corr = np.zeros((nblocks, tinblock), dtype=np.complex_)
            for i in range(0, tinblock, int(tinblock / 10)):
                corr[t] += np.array(autocorr(enkcorr[j, (tblock * t +i):(tblock * t + tinblock +i), 0])) / 10 / 3
                corr[t] += np.array(autocorr(enkcorr[j, (tblock * t +i):(tblock * t + tinblock +i), 1])) / 10 / 3
                corr[t] += np.array(autocorr(enkcorr[j, (tblock * t +i):(tblock * t + tinblock +i), 2])) / 10 / 3

            chik = (np.var(enkcorr[j, :, 0])  + np.var(enkcorr[j, :, 1]) + np.var(enkcorr[j, :, 2])) / 3

            ft[j - 1] = chik / (np.cumsum(corr[t, :int(tinblock / 2) + 1]) * (2 * (j) * np.pi / L) ** 2) * (
                        fac / dt * (1e-10) ** 2 / 1e-12)
            corrk[t, j-1] = corr[t]
            #print(chik, corr[t,0])

        corren[t] = ft
    return corren, chi, corrk

