from modules import initialize
from modules import dipole
import numpy as np
from scipy import signal
from scipy.fft import fft
from modules import compute
import matplotlib.pyplot as plt
import os


def autocorr(x):
    result = signal.correlate(x, x, mode='full', method='fft')
    v = [result[i] / (len(x) - abs(i - (len(x)) + 1)) for i in range(len(result))]
    return np.array(v[int(result.size / 2):])


def computenltt(root, filename, Np, L, posox, nk, ntry, temp):
    print(root, filename, Np, L, posox, nk, ntry, temp)
    mantaindata = True
    plot = False
    if os.path.exists(root + 'enk.npy') and mantaindata:
        enk = np.load(root + 'enk.npy')
        dipenkx = np.load(root + 'dipenkx.npy')
        dipenky = np.load(root + 'dipenky.npy')
        chk = np.load(root + 'chk.npy')
        dipkx = np.load(root + 'dipkx.npy')
        dipky = np.load(root + 'dipky.npy')
        nsnap = np.shape(enk)[1] / 3
        with open(root + 'output.out', 'a') as g:
            print('number of total snapshots is', nsnap)
            print('done')
            g.write('number of total snapshots is' + '{}\n'.format(nsnap))
            g.write('done')
    else:
        nsnap, enk, dipenkx, dipenky, chk, dipkx, dipky = compute.computekft(root, filename, Np, L, posox, nk, ntry)

    ndata = int(enk.shape[1])

    enkcorr = np.reshape(enk, (nk, int(ndata / 3), 3))

    nblocks = 10

    tblock = int(enkcorr.shape[1] / nblocks)

    tinblock = int(tblock / 2)

    rho = Np / (6.022e23 * L ** 3 * 1.e-30)  # mol/m^3

    cp = 18.0e-3  # Kcal/mol*k

    fac = 4186 * rho * cp  # J/k/m^3

    dt = 0.5  # fs

    tdump = 20  # dump step

    dt = 0.5 * tdump  # fs

    corr = np.zeros((nblocks, tinblock), dtype=np.complex_)

    chi = np.var(enk[:, :], axis=1)  # (Kcal*m)**2

    nk = 10

    corren = np.zeros((nblocks, nk - 1, int(tinblock / 2) + 1), dtype=np.complex_)

    ft = np.zeros((nk - 1, int(tinblock / 2) + 1), dtype=np.complex_)

    for t in range(nblocks):

        print(t)

        for j in range(1, nk):

            for i in range(0, tinblock, int(tinblock / 100)):
                corr[t] += np.array(autocorr(enkcorr[j, (tblock * t + i):(tblock * t + tinblock + i), 0])) / 100 / 3
                corr[t] += np.array(autocorr(enkcorr[j, (tblock * t + i):(tblock * t + tinblock + i), 1])) / 100 / 3
                corr[t] += np.array(autocorr(enkcorr[j, (tblock * t + i):(tblock * t + tinblock + i), 2])) / 100 / 3

            chik = (np.var(enkcorr[j, :, 0]) + np.var(enkcorr[j, :, 1]) + np.var(enkcorr[j, :, 2])) / 3

            ft[j - 1] = chik / (np.cumsum(corr[t, :int(tinblock / 2) + 1]) * (2 * (j) * np.pi / L) ** 2) * (
                        fac / dt * (1e-10) ** 2 / 1e-15)

        corren[t] = ft
    return corren

