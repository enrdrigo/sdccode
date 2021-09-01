from modules import initialize
from modules import dipole
import numpy as np
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt
import os


def stdblock(array):

    var = list()
    binsize = list()
    nbino = 0
    for i in range(1, int(len(array))+1):

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


root = '../'
filename = 'dump1.1fs.lammpstrj'
posox = 0.125
L, Linf = initialize.getBoxboundary(filename, root)
Npart = initialize.getNpart(filename, root)
filebin = filename + '{}'.format(Npart)+'.npy'
print('lato cella', L)
f = open('file.out', '+w')
f.write('lato cella'+'{}\n'.format(L))
f.close()


if os.path.exists(root+filebin):

    dati = np.load(root+filebin)

else:

    dati = initialize.getdatafromfile(filename, root, Npart)

nsnap = initialize.getNsnap(dati, Npart)


g = open('file.out', 'a')
g.write('got nsnap\n')
g.close()

nk = 10
q = np.zeros((nk, nsnap), dtype=np.complex_)
q1 = np.zeros((nk, nsnap), dtype=np.complex_)
print('calcolo dipolo')

for j in range(nsnap):

    if j % int(nsnap/10) == 0:

        print('#{}'.format(int(j/nsnap * 100)+1)+'% ', end='', flush=True)

    datisnap = np.array(dati[j * Npart: (j + 1) * Npart])

    poschO, posO, posH1, posH2 = dipole.computeposmol(Npart, L, Linf, nsnap, datisnap.transpose(), posox)

    en_at, pos_at, em, endip = dipole.computeaten(Npart, L, Linf, nsnap, datisnap.transpose(), poschO, posH1, posH2)

    for i in range(nk):

        q1[i][j] = np.sum(en_at[:]*np.exp(-1j*pos_at[:, 0]*2*i*np.pi/L))

print('')
print('fine calcolo dipolo')
qm = np.sum(q1, axis=1)/nsnap
q = np.transpose(np.transpose(q1)-qm)
f = open('nltt'+'{}'.format(Npart)+'.dat', 'w+')
print('inizio calcolo non local thermal conductivity:')
told = 0
nlttlist = []
rho = Npart/(6.022e23 * L ** 3 * 1.e-30)
cp = 18.0e-3
fac = 4.186*rho*cp
nltt = []
sdd = []
print('#{}'.format(1) + '% ', end='', flush=True)
taumax = int(nsnap/10)

for s in range(1, taumax):

    nltts = np.zeros((int(nsnap/s), nk - 1))
    corr = np.zeros((nk, s), dtype=np.complex_)
    result = np.zeros(2 * s, dtype=np.complex_)

    if s % int(taumax / 10) == 0:

        print('#{}'.format(int(s / taumax * 100)) + '% ', end='', flush=True)

    if int(nsnap/s) == told:

        continue

    told = int(nsnap/s)

    for t in range(int(nsnap/s)):

        for j in range(nk):

            result = signal.correlate((q[j, s*t:s*(t+1)]), q[j, s*t:s*(t+1)],  mode='full', method='fft')
            v = [result[i] / (len(q[j, s*t:s*(t+1)])-abs(i - (len(q[j, s*t:s*(t+1)])) + 1)) for i in range(len(result))]
            corr[j] = np.array(v[int(result.size/2):])

        ft = np.zeros((nk, int(s/5)+1), dtype=np.complex_)
        for i in range(nk):

            ft[i] = fft(np.real(corr[i][: int(s/5)+1]))

        chi = np.var(q1[:, s*t:s*(t+1)], axis=1)
        for i in range(1, nk):

            nltts[t, i-1] = np.real(chi[i]/(ft[i, 0]*(2*i*np.pi/L)**2)*1.e-5*fac)

    nlttlist.append(nltts)

    mnltt = np.mean(nltts, axis=0)

    nltt.append(mnltt)

    ssdd = np.zeros(nk-1)
    for i in range(nk-1):
        var, xbin = stdblock(nltts[:, i])

        index = int(len(var)/4)

        ssdd[i] = np.sqrt(np.mean(var[index: -index-1]))  # np.sqrt(np.var(nltts[:, i]) / int(nsnap / s))

    sdd.append(ssdd)

    f.write('{}\t'.format(s))
    f.write('{}\t'.format(mnltt[0]) + '{}\t'.format(ssdd[0]))
    f.write('{}\t'.format(mnltt[1]) + '{}\t'.format(ssdd[1]))
    f.write('{}\t'.format(mnltt[2]) + '{}\t'.format(ssdd[2]))
    f.write('{}\t'.format(mnltt[3]) + '{}\n'.format(ssdd[3]))

print('#100% ')
print('Done')
f.close()


nlttmean=np.zeros((len(nltt), nk-1))

for i in range(len(nltt)):
    nlttmean[i]=nltt[i]


with open('nlttk'+'{}'.format(Npart)+'.dat', 'w+') as f:

    for i in range(nk-1):
        q = int(len(nltt)/2)

        qq = int(len(nltt)/4)

        f.write('{}\t'.format((i+1)*2*np.pi/L)+'{}\t'.format(np.mean(nlttmean[qq:-qq-1, i]))+'{}\n'.format(sdd[q][i]))

nlttarray = np.array(nlttlist, dtype=object)

for i in range(65, 70):

    var, xbin = stdblock(nlttarray[i][:, 0])

    plt.plot(xbin, np.sqrt(var), label='{}'.format(i))

plt.xlabel('bin lenght')
plt.ylabel(r'$\sigma_b$')
plt.legend()
plt.show()


