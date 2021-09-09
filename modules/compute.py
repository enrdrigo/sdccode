import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE POSITION OF THE OXY AND OF THE TWO HYDROGENS AT GIVEN SNAPSHOT. IT ALSO GETS THE POSITION OF THE
# FOURTH PARTICLE IN THE TIP4P/2005 MODEL OF WATER WHERE THERE IS THE CHARGE OF THE OXY (SEE TIP4P/2005 MODEL OF WATER).


def computeposmol(Np, data_array, posox):
    nmol = int(Np / 3)
    datamol = np.zeros((8, nmol, 3))
    datamol = data_array.reshape((8, nmol, 3))

    #
    posO = np.zeros((3, nmol))
    posH1 = np.zeros((3, nmol))
    posH2 = np.zeros((3, nmol))
    posO[0] = np.transpose(datamol[2])[0]
    posO[1] = np.transpose(datamol[3])[0]
    posO[2] = np.transpose(datamol[4])[0]
    posH1[0] = np.transpose(datamol[2])[1]
    posH1[1] = np.transpose(datamol[3])[1]
    posH1[2] = np.transpose(datamol[4])[1]
    posH2[0] = np.transpose(datamol[2])[2]
    posH2[1] = np.transpose(datamol[3])[2]
    posH2[2] = np.transpose(datamol[4])[2]
    #

    #
    bisdir = np.zeros((3, nmol))
    bisdir = 2 * posO - posH1 - posH2
    #

    #
    poschO = np.zeros((3, nmol))
    poschO = posO - posox * bisdir / np.sqrt(bisdir[0] ** 2 + bisdir[1] ** 2 + bisdir[2] ** 2)
    #

    return poschO, posO, posH1, posH2


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE MOLECULAR DIPOLES AND THE CENTER OF MASS OF THE MOLECULES AT GIVEN SNAPSHOT. COMPITING THE MOLECULAR
# DIPOLE WE MUST REMEMBER THAT THE OXY CHARGE IS NOT LOCATED IN THE OXY POSITION (SEE TIP4P/2005 MODEL OF WATER).


def computemol(Np, data_array, poschO, posO, posH1, posH2):
    nmol = int(Np / 3)
    datamol = np.zeros((8, nmol, 3))
    datamol = data_array.reshape((8, nmol, 3))
    #

    #
    chO = np.zeros(nmol)
    chH1 = np.zeros(nmol)
    chH2 = np.zeros(nmol)
    chO = np.transpose(datamol[5])[0]
    chH1 = np.transpose(data_array[5])[1]
    chH2 = np.transpose(data_array[5])[2]
    #

    #
    cdmmol = np.zeros((3, nmol))
    cdmmol = (posO * 15.9994 + (posH1 + posH2) * 1.008) / (15.9994 + 2 * 1.008)
    #

    #
    pos_mch = np.zeros((3, nmol))
    pos_mch = poschO * chO + posH1 * chH1 + posH2 * chH2
    #

    #
    dip_mol0 = np.zeros((3, nmol))
    dip_mol0 = pos_mch
    #

    return np.transpose(dip_mol0), np.transpose(cdmmol)


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE CHARGE AND ATOMIC POSITION ARRAYS OF THE ATOMS AT A GIVEN SNAPSHOT. THE OXY POSITION IS SHIFTED ACCORDING
# TO THE TIP4P/2005 MODEL OF WATER.


def computeat(Np, data_array, poschO, posH1, posH2):
    nmol = int(Np / 3)

    #
    chat = np.zeros(Np)
    chat = data_array[5]
    #

    #
    posm = np.zeros((3, nmol, 3))
    posm[0] = np.transpose(poschO)
    posm[1] = np.transpose(posH1)
    posm[2] = np.transpose(posH2)
    #

    #
    pos_at = np.zeros((3, Np))
    test = posm.transpose()
    pos_at = test.reshape((3, Np))
    #

    # THIS IS IN FACT THE CHARGE TIMES A PHASE WHERE GAT = 2 * np.pi * np.array((1e-8, 1e-8, 1e-8)) / L. I DO THIS
    # IN ORDER TO COMPUTE PROPERLY THE STATIC DIELECTRIC CONSTANT VIA THE FOURIER TRANFORM OR THE CHARGE OVER
    # THE MODULUS OF G,  AT G \APPROX 0
    ch_at = np.zeros(Np)
    ch_at = chat
    #

    return ch_at, np.transpose(pos_at)


def computeaten(Np, data_array, poschO, posH1, posH2):
    nmol = int(Np / 3)
    #
    datamol = np.zeros((8, nmol, 3))
    datamol = data_array.reshape((8, nmol, 3))
    #
    en0 = np.zeros(nmol)
    enH1 = np.zeros(nmol)
    enH2 = np.zeros(nmol)
    enO = np.transpose(datamol[6])[0] + np.transpose(datamol[7])[0]
    enH1 = np.transpose(datamol[6])[1] + np.transpose(datamol[7])[1]
    enH2 = np.transpose(datamol[6])[2] + np.transpose(datamol[7])[2]

    #
    enat = np.zeros(Np)
    enat = data_array[6] + data_array[7]
    #

    #
    pos_at = np.zeros((3, Np))
    pos_at[0] = data_array[2]
    pos_at[1] = data_array[3]
    pos_at[2] = data_array[4]
    #

    #
    en_at = np.zeros(Np)
    en_at = enat
    #

    endip = np.zeros((3, nmol))
    endip = poschO*(enO-np.sum(enat)/Np) + posH1*(enH1-np.sum(enat)/Np) + posH2*(enH2-np.sum(enat)/Np)

    return en_at, np.transpose(pos_at), np.sum(enat), np.transpose(endip)


def computekft(root, filename, Np, L, posox, nk, ntry):
    enk = []
    dipenkx = []
    dipenky = []
    chk = []
    dipkx = []
    dipky = []
    with open(root + 'output.out', 'a') as g:
        print('start the computation of the fourier transform of the densities')
        g.write('start the computation of the fourier transform of the densities\n')
    with open(root+filename, 'r') as f:
        line = f.readline()
        while line != '':

            d = []
            for p in range(Np+9):

                if len(line.split(' ')) != 8:
                    line = f.readline()
                    continue
                dlist = [float(x.strip('\n')) for x in line.split(' ')]
                line = f.readline()
                d.append(dlist)

            datisnap = np.array(d)

            poschO, posO, posH1, posH2 = computeposmol(Np, datisnap.transpose(), posox)

            dip_mol, cdmol = computemol(Np, datisnap.transpose(), poschO, posO, posH1, posH2)

            ch_at, pos_at = computeat(Np, datisnap.transpose(), poschO, posH1, posH2)

            en_at, posatomic, em, endip = computeaten(Np, datisnap.transpose(), posO, posH1, posH2)

            emp = em/Np*np.ones(Np)

            enklist = [np.sum((en_at[:] - emp[:]) * np.exp(1j * posatomic[:, 0] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            dipenkxlist = [np.sum((endip[:,  0]) * np.exp(1j * cdmol[:, 0] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            dipenkylist = [np.sum((endip[:, 1]) * np.exp(1j * cdmol[:,  0] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            chklist = [np.sum((ch_at[:]) * np.exp(-1j * pos_at[:, 0] * 2 * (i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            dipkxlist = [np.sum((dip_mol[:, 0]) * np.exp(1j * cdmol[:,  0] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            dipkylist = [np.sum((dip_mol[:, 1]) * np.exp(1j * cdmol[:, 0] * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L), axis=0) for i in range(nk)]

            enk.append(enklist)

            dipenkx.append(dipenkxlist)

            dipenky.append(dipenkylist)

            chk.append(chklist)

            dipkx.append(dipkxlist)

            dipky.append(dipkylist)

            with open(root + 'output.out', 'a') as g:
                if len(chk)%2000 == 0:
                    print('got '+str(len(chk))+' snapshot')
                    g.write('got '+str(len(chk))+' snapshot\n')

            if len(chk) == ntry:
                return len(chk), np.transpose(np.array(enk)), np.transpose(np.array(dipenkx)), np.transpose(np.array(dipenky)), np.transpose(np.array(chk)), np.transpose(np.array(dipkx)), np.transpose(np.array(dipky))
        with open(root + 'output.out', 'a') as g:
            print('number of total snapshots is', len(chk))
            print('done')
            g.write('number of total snapshots is'+'{}\n'.format(len(chk)))
            g.write('done')

        np.save(root+'enk.npy', np.transpose(np.array(enk)))
        np.save(root+'dipenkx.npy', np.transpose(np.array(dipenkx)))
        np.save(root+'dipenky.npy', np.transpose(np.array(dipenky)))
        np.save(root+'chk.npy', np.transpose(np.array(chk)))
        np.save(root+'dipkx.npy', np.transpose(np.array(dipkx)))
        np.save(root+'dipky.npy', np.transpose(np.array(dipky)))
        return len(chk), np.transpose(np.array(enk)), np.transpose(np.array(dipenkx)), np.transpose(np.array(dipenky)), np.transpose(np.array(chk)), np.transpose(np.array(dipkx)), np.transpose(np.array(dipky))


def stdblock(array):
    var = list()
    binsize = list()
    nbino = 0
    for i in range(1, 500):
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


def computestaticresponse(root, filename, Np, L, posox, nk, ntry, temp):
    if os.path.exists(root+'enk.npy'):
        enk = np.load(root+'enk.npy')
        dipenkx = np.load(root + 'dipenkx.npy')
        dipenky = np.load(root + 'dipenky.npy')
        chk = np.load(root + 'chk.npy')
        dipkx = np.load(root + 'dipkx.npy')
        dipky = np.load(root + 'dipky.npy')
        nsnap = np.shape(enk)[1]
        with open(root + 'output.out', 'a') as g:
            print('number of total snapshots is', nsnap)
            print('done')
            g.write('number of total snapshots is'+'{}\n'.format(nsnap))
            g.write('done')
    else:
        nsnap, enk, dipenkx, dipenky, chk, dipkx, dipky = computekft(root, filename, Np, L, posox, nk, ntry)

    fac = (16.022 * 1.0e-30 * 4184 / 6.02214e23 * 1.0e-10 / (L ** 3 * 1.0e-30 * 1.38e-23 * temp * 8.854 * 1.0e-12))
    face = (16.022**2) * 1.0e5 / (L**3 * 1.38 * temp * 8.854)

    xk = np.linspace(0, nk - 1, nk) * 2 * np.pi / L + np.sqrt(3.) * 1.0e-5 * 2 * np.pi / L
    a = np.zeros(nk, np.complex_)
    b = np.zeros(nk, np.complex_)
    c = np.zeros(nk, np.complex_)
    d = np.zeros(nk, np.complex_)
    va = np.zeros(nk)
    vb = np.zeros(nk)
    vc = np.zeros(nk)
    vd = np.zeros(nk)

    for i in range(nk):
        a[i] = np.mean((enk[i] / xk[i]) * np.conj(chk[i] / xk[i])) * fac
        b[i] = np.mean(dipenkx[i] * np.conj(dipkx[i])) * fac
        c[i] = np.mean(dipenky[i] * np.conj(dipky[i])) * fac
        d[i] = np.mean((chk[i] / xk[i]) * np.conj(chk[i] / xk[i])) * face

    for i in range(nk):
        std, bins = np.sqrt(stdblock((enk[i] / xk[i]) * np.conj(chk[i] / xk[i]) * fac))
        pp = int(len(std) / 2)
        va[i] = std[pp]
        std, bins = np.sqrt(stdblock(dipenkx[i] * np.conj(dipkx[i]) * fac))
        vb[i] = std[pp]
        std, bins = np.sqrt(stdblock(dipenky[i] * np.conj(dipky[i]) * fac))
        vc[i] = std[pp]
        std, bins = np.sqrt(stdblock((chk[i] / xk[i]) * np.conj(chk[i] / xk[i]) * face))
        vd[i] = std[pp]
    with open(root+'staticresponse.out', '+w') as g:
        g.write('#k\t chtpc\t dipxxtpc\t dipyytpc\t chdiel\n')
        for i in range(nk):
            g.write('{}\t'.format(xk[i]))
            g.write('{}\t'.format(np.real(a[i])) + '{}\t'.format(np.real(va[i])))
            g.write('{}\t'.format(np.real(b[i])) + '{}\t'.format(np.real(vb[i])))
            g.write('{}\t'.format(np.real(c[i])) + '{}\t'.format(np.real(vc[i])))
            g.write('{}\t'.format(np.real(d[i])) + '{}\n'.format(np.real(vd[i])))

    fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
    plt.errorbar(xk[0:], a[0:], va, fmt='.-', label=r'$\langle\frac{\rho(k)\left(e(-k)-e(0)\right)}{k^2}\rangle$')
    plt.errorbar(xk[0:], b[0:], vb, fmt='.-', label=r'$\langle p_{energy_{x}}(k)p_{charge_{x}}(-k)\rangle$')
    plt.errorbar(xk[0:], c[0:], vc, fmt='.-', label=r'$\langle p_{energy_{y}}(k)p_{charge_{y}}(-k)\rangle$')
    plt.xlabel(r'k ($\AA^{-1}$)')
    plt.ylabel(r'$\frac{P}{\epsilon_0\triangledown (T)/T }$ (V)')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
    plt.errorbar(xk[0:], d[0:], vd, fmt='.-', label=r'$\langle\frac{\rho(k)\rho(-k)}{k^2}\rangle$')
    plt.xlabel(r'k ($\AA^{-1}$)')
    plt.ylabel(r'$\epsilon_r$')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
    v, x = stdblock((enk[0] / xk[0]) * np.conj(chk[0] / xk[0]) * fac)
    plt.plot(x, np.sqrt(v))
    plt.ylabel(r'$\sigma_b$ of $\langle\frac{\rho(k_{min})\left(e(-k_{min})-e(0)\right)}{k_{min}^2}\rangle$')
    plt.xlabel('block size')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(8, 6), constrained_layout=True)
    v, x = stdblock((chk[0] / xk[0]) * np.conj(chk[0] / xk[0]) * face)
    plt.plot(x, np.sqrt(v))
    plt.ylabel(r'$\sigma_b$ of $\langle\frac{\rho(k_{min})\rho(-k_{min})}{k_{min}^2}\rangle$')
    plt.xlabel('block size')
    plt.show()

    out = {}

    out['dielectric'] = {}

    out['thermopolarization'] = {}

    out['dielectric']['charge'] = {'mean': d, 'std': vd}

    out['thermopolarization']['charge'] = {'mean': a, 'std': va}

    out['thermopolarization']['dipole']={}

    out['thermopolarization']['dipole']['xx'] = {'mean': b, 'std': vb}

    out['thermopolarization']['dipole']['yy'] = {'mean': c, 'std': vc}

    return out