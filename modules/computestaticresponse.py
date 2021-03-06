import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os


def stdblock(array):
    var = list()
    binsize = list()
    nbino = 0
    for i in range(1, int(len(array)/10)):
        nbin = int(len(array) / i)
        if nbin == nbino:
            continue
        rarray = np.reshape(array[:nbin * i], (nbin, i))
        barray = np.zeros(nbin, dtype=np.complex_)
        barray = np.sum(rarray, axis=1) / i
        var.append(np.var(barray) / nbin)
        binsize.append(i)
        nbino = nbin

    return np.array(var), np.array(binsize)

def computestaticresponse(root, L, nk, temp):
    plot = False

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
        nsnap = int(len(chkb) / 3)
    else:
        raise ValueError

    fac = (16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / (L ** 3 * 1.0e-30 * 1.38e-23 * temp * 8.854 * 1.0e-12))
    face = (16.022 ** 2) * 1.0e5 / (L ** 3 * 1.38 * temp * 8.854)

    xk = np.linspace(0, nk - 1, nk) * 2 * np.pi / L + np.sqrt(3.) * 1.0e-5 * 2 * np.pi / L
    a = np.zeros(nk, np.complex_)
    b = np.zeros(nk, np.complex_)
    c = np.zeros(nk, np.complex_)
    d = np.zeros(nk, np.complex_)
    e = np.zeros(nk, np.complex_)

    va = np.zeros(nk)
    vb = np.zeros(nk)
    vc = np.zeros(nk)
    vd = np.zeros(nk)
    ve = np.zeros(nk)

    with open(root + 'enk.out', '+w') as g:
        for i in range(nk):
            g.write('{}\t'.format(xk[i]) + '{}\n'.format(np.abs(np.mean(enk[i]))))

    for i in range(nk):
        a[i] = np.mean((enk[i] / xk[i]) * np.conj(chk[i] / xk[i])) * fac
        b[i] = np.mean(dipenkx[i] * np.conj(dipkx[i])) * fac
        c[i] = np.mean(dipenky[i] * np.conj(dipky[i])) * fac
        d[i] = np.mean((chk[i] / xk[i]) * np.conj(chk[i] / xk[i])) * face
        e[i] = np.mean(dipkx[i] * np.conj(dipkx[i])) * face

    convergence1 = np.real((np.cumsum((enk[0][:] / xk[0]) * np.conj(chk[0][:] / xk[0])) * fac) / (
                np.cumsum((chk[0][:] / xk[0]) * np.conj(chk[0][:] / xk[0])) * face) / temp)
    convergence2 = np.real((np.cumsum((enk[1][:] / xk[1]) * np.conj(chk[1][:] / xk[1])) * fac) / (
                np.cumsum((chk[1][:] / xk[1]) * np.conj(chk[1][:] / xk[1])) * face) / temp)
    convergence3 = np.real((np.cumsum((enk[2][:] / xk[2]) * np.conj(chk[2][:] / xk[2])) * fac) / (
                np.cumsum((chk[2][:] / xk[2]) * np.conj(chk[2][:] / xk[2])) * face) / temp)
    convergence4 = np.real((np.cumsum((enk[3][:] / xk[3]) * np.conj(chk[3][:] / xk[3])) * fac) / (
                np.cumsum((chk[3][:] / xk[3]) * np.conj(chk[3][:] / xk[3])) * face) / temp)
    #  a/d/temp
    with open(root + 'convergence.out', '+w') as g:
        for i in range(1, len(enk[0]), 10):
            # convergence1 = np.real((np.mean((enk[0][:i] / xk[0]) * np.conj(chk[0][:i] / xk[0])) * fac)/(np.mean((chk[0][:i] / xk[0]) * np.conj(chk[0][:i] / xk[0])) * face)/temp)
            # convergence3 = np.real((np.mean((enk[1][:i] / xk[1]) * np.conj(chk[1][:i] / xk[1])) * fac)/(np.mean((chk[1][:i] / xk[1]) * np.conj(chk[1][:i] / xk[1])) * face)/temp)
            # convergence2 = np.real((np.mean((enk[2][:i] / xk[2]) * np.conj(chk[2][:i] / xk[2])) * fac)/(np.mean((chk[2][:i] / xk[2]) * np.conj(chk[2][:i] / xk[2])) * face)/temp)
            # convergence4 = np.real((np.mean((enk[3][:i] / xk[3]) * np.conj(chk[3][:i] / xk[3])) * fac)/(np.mean((chk[3][:i] / xk[3]) * np.conj(chk[3][:i] / xk[3])) * face)/temp)
            g.write('{}\t'.format(i) + '{}\t'.format(convergence1[i]) + '{}\t'.format(convergence2[i]) + '{}\t'.format(
                convergence3[i]) + '{}\n'.format(convergence4[i]))

    for i in range(nk):
        std, bins = np.sqrt(stdblock((enk[i] / xk[i]) * np.conj(chk[i] / xk[i]) * fac))
        pp = int(19 * len(std) / 20)
        va[i] = std[pp]
        std, bins = np.sqrt(stdblock(dipenkx[i] * np.conj(dipkx[i]) * fac))
        vb[i] = std[pp]
        std, bins = np.sqrt(stdblock(dipenky[i] * np.conj(dipky[i]) * fac))
        vc[i] = std[pp]
        std, bins = np.sqrt(stdblock((chk[i] / xk[i]) * np.conj(chk[i] / xk[i]) * face))
        vd[i] = std[pp]
        std, bins = np.sqrt(stdblock(dipkx[i] * np.conj(dipkx[i]) * face))
        ve[i] = std[pp]

    with open(root + 'staticresponse.out', '+w') as g:
        g.write('#k\t chtpc\t dipxxtpc\t dipyytpc\t chdiel\t dipxxdiel\n')
        for i in range(nk):
            g.write('{} \t'.format(xk[i]))
            g.write('{} \t'.format(np.real(a[i])) + '{} \t'.format(np.real(va[i])))
            g.write('{} \t'.format(np.real(b[i])) + '{} \t'.format(np.real(vb[i])))
            g.write('{} \t'.format(np.real(c[i])) + '{} \t'.format(np.real(vc[i])))
            g.write('{} \t'.format(np.real(d[i])) + '{} \t'.format(np.real(vd[i])))
            g.write('{} \t'.format(np.real(e[i])) + '{} \n'.format(np.real(ve[i])))

    v, x = stdblock((chk[1] / xk[1]) * np.conj(chk[1] / xk[1]) * face)
    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.plot(x, np.sqrt(v))
        plt.ylabel(r'$\sigma_b$ of $\langle\frac{\rho(k_{min})\rho(-k_{min})}{k_{min}^2}\rangle$')
        plt.xlabel('block size')
        plt.show(block=False)

    with open(root + 'blockanalisisvardckmin.out', 'w+') as g:
        for i in range(len(v)):
            g.write('{}\t'.format(x[i]) + '{}\n'.format(np.sqrt(v[i])))

    v, x = stdblock((enk[1] / xk[1]) * np.conj(chk[1] / xk[1]) * fac)
    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.plot(x, np.sqrt(v))
        plt.ylabel(r'$\sigma_b$ of $\langle\frac{\rho(k_{min})\left(e(-k_{min})-e(0)\right)}{k_{min}^2}\rangle$')
        plt.xlabel('block size')
        plt.show(block=False)

    with open(root + 'blockanalisisvartpckmin.out', 'w+') as g:
        for i in range(len(v)):
            g.write('{}\t'.format(x[i]) + '{}\n'.format(np.sqrt(v[i])))

    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.errorbar(xk[0:], d[0:], vd, fmt='.-', label=r'$\langle\frac{\rho(k)\rho(-k)}{k^2}\rangle$')
        plt.errorbar(xk[0:], e[0:], ve, fmt='.-', label=r'$\langle\frac{p_{charge_x}(k)p_{charge_x}(-k)}{k^2}\rangle$')
        plt.xlabel(r'k ($\AA^{-1}$)')
        plt.ylabel(r'$\epsilon_r$')
        plt.legend()
        plt.show(block=False)

        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.errorbar(xk[0:], a[0:], va, fmt='.-', label=r'$\langle\frac{\rho(k)\left(e(-k)-e(0)\right)}{k^2}\rangle$')
        plt.errorbar(xk[0:], b[0:], vb, fmt='.-', label=r'$\langle p_{energy_{x}}(k)p_{charge_{x}}(-k)\rangle$')
        plt.errorbar(xk[0:], c[0:], vc, fmt='.-', label=r'$\langle p_{energy_{y}}(k)p_{charge_{y}}(-k)\rangle$')
        plt.xlabel(r'k ($\AA^{-1}$)')
        plt.ylabel(r'$\frac{P}{\epsilon_0\triangledown (T)/T }$ (V)')
        plt.legend()
        plt.show(block=False)

    stdch = np.real(np.sqrt((va /(1 -1/d[0]) / temp) ** 2))# + (a / d ** 2 / temp * vd) ** 2))
    tpcch = np.real(a / (1 - 1 / d[0]) / temp)

    stddip = np.real(np.sqrt((vb / e / temp) ** 2 + (b / e ** 2 / temp * ve) ** 2))
    tpcdip = np.real(b / e / temp)

    if plot:
        fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
        plt.errorbar(xk, tpcch, stdch, fmt='.-', label='computed via the charges')
        plt.errorbar(xk, tpcdip, stddip, fmt='.-', label='computed via the dipoles')
        plt.xlabel(r'k ($\AA^{-1}$)')
        plt.ylabel(r'$\frac{E}{\triangledown (T) }$ (V/K)')
        plt.legend()
        plt.show(block=False)

    with open(root + 'thermopolarizationresponse.out', '+w') as g:
        g.write('# k\t tpc via the charge \t tpc via the dipoles\n')
        for i in range(1,nk):
            g.write('{}\t'.format(xk[i]))
            g.write('{}\t'.format(tpcch[i]) + '{}\t'.format(stdch[i]))
            g.write('{}\t'.format(tpcdip[i]) + '{}\n'.format(stddip[i]))
    if plot:
        plt.show()

    out = dict()

    out['dielectric'] = dict()

    out['thermopolarization'] = dict()

    out['dielectric']['charge'] = {'mean': d, 'std': vd}

    out['dielectric']['dipole'] = dict()

    out['dielectric']['dipole']['xx'] = {'mean': e, 'std': ve}

    out['thermopolarization']['charge'] = {'mean': a, 'std': va}

    out['thermopolarization']['dipole'] = dict()

    out['thermopolarization']['dipole']['xx'] = {'mean': b, 'std': vb}

    out['thermopolarization']['dipole']['yy'] = {'mean': c, 'std': vc}

    return out
