from modules import initialize
from modules import dipole
import numpy as np
from scipy import signal
from scipy.fft import fft
from modules import compute
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


root = './'
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
    np.save(root+filebin, dati)

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

    poschO, posO, posH1, posH2 = dipole.computeposmol(Npart, datisnap.transpose(), posox)

    en_at, pos_at, em, endip = dipole.computeaten(Npart, datisnap.transpose(), poschO, posH1, posH2)

    for i in range(nk):

        q1[i][j] = np.sum(en_at[:]*np.exp(-1j*pos_at[:, 0]*2*i*np.pi/L))

print('')
print('fine calcolo dipolo')
qm = np.sum(q1, axis=1)/nsnap
q = np.transpose(np.transpose(q1)-qm)
f = open('nltt'+'{}'.format(Npart)+'.dat', 'w+')
timestep = float(input('timestep:\n'))
dumpstep = float(input('dumpstep:\n'))
dt = timestep*dumpstep
print('inizio calcolo non local thermal conductivity:')
told = 0
nlttlist = []
rho = Npart/(6.022e23 * L ** 3 * 1.e-30)  # mol/m^3
cp = 18.0e-3  # Kcal/mol*k
fac = 4186*rho*cp  # J/k/m^3
nltt = []
sdd = []
print('#{}'.format(1) + '% ', end='', flush=True)
taumax = int(nsnap/2)

tmax = 400

sl = []

nl0 = []

sl0 = []

for s in range(tmax*2, taumax):

    corr = np.zeros((nk, s), dtype=np.complex_)
    result = np.zeros(2 * s, dtype=np.complex_)

    if (s - tmax*2 + 1) % int((taumax - tmax*2) / 10) == 0:

        print('#{}'.format(int((s - tmax*2 + 1) / (taumax - tmax*2) * 100)) + '% ', end='', flush=True)

    if int(nsnap/s) == told:

        continue

    told = int(nsnap/s)

    nltts = np.zeros((int(nsnap / s), nk - 1))

    for t in range(int(nsnap/s)):

        for j in range(nk):

            result = signal.correlate((q[j, s*t:s*(t+1)]), q[j, s*t:s*(t+1)],  mode='full', method='fft')
            v = [result[i] / (len(q[j, s*t:s*(t+1)])-abs(i - (len(q[j, s*t:s*(t+1)])) + 1)) for i in range(len(result))]
            corr[j] = np.array(v[int(result.size/2):])


        ft = np.zeros(nk, dtype=np.complex_)
        for i in range(nk):

            ft[i] = np.sum(corr[i, :min(int(s/2), tmax)+1]) * dt   # fft(np.real(corrnew[i][: int(int(s/av)/5)+1])) * dt * av # (Kcal*m)**2 * fs

        chi = np.var(q1[:, :], axis=1)  # (Kcal*m)**2
        for i in range(1, nk):

            nltts[t, i-1] = np.real(chi[i]/(ft[i, 0]*(2*i*np.pi/L)**2)*fac) *(1e-10)**2/1e-15  # m^2/s * J/k*m^3 = J/s /m

    nlttlist.append(nltts)

    mnltt = np.mean(nltts, axis=0)

    nltt.append(mnltt)

    ssdd = np.zeros(nk-1)
    for i in range(nk-1):
        var, xbin = stdblock(nltts[:, i])

        index = int(len(var)/4)

        ssdd[i] = np.sqrt(np.mean(var[index: -index-1]))  # np.sqrt(np.var(nltts[:, i]) / int(nsnap / s))

    sdd.append(ssdd)

    sl.append(s)

    nl0.append(mnltt[0])

    sl0.append(ssdd[0])

    f.write('{}\t'.format(s))
    f.write('{}\t'.format(mnltt[0]) + '{}\t'.format(ssdd[0]))
    f.write('{}\t'.format(mnltt[1]) + '{}\t'.format(ssdd[1]))
    f.write('{}\t'.format(mnltt[2]) + '{}\t'.format(ssdd[2]))
    f.write('{}\t'.format(mnltt[3]) + '{}\n'.format(ssdd[3]))

print('#100% ')
print('Done')
f.close()


nlttmean = np.zeros((len(nltt), nk-1))

for i in range(len(nltt)):
    nlttmean[i] = nltt[i]

sdd = np.array(sdd)

print(len(nlttmean[:, 6]), len(sdd[:, 6]))

with open('nlttk'+'{}'.format(Npart)+'.dat', 'w+') as f:

    xk = []

    nlk = []

    slk = []

    for i in range(nk-1):
        p = int(len(nltt)/2)

        pp = int(len(nltt)/4)

        var = stdblock(nlttmean[pp:, i])[0]

        std2mean = np.mean(sdd[pp:][i]**2)/len(sdd[pp:][i]) + var[-2]

        f.write('{}\t'.format((i+1)*2*np.pi/L)+'{}\t'.format(np.mean(nlttmean[p:, i]))+'{}\n'.format(np.mean(sdd[p:, i])))

        xk.append((i+1)*2*np.pi/L)

        nlk.append(np.mean(nlttmean[p:, i]))

        slk.append(np.mean(sdd[p:, i]))

nlttarray = np.array(nlttlist, dtype=object)


plt.errorbar(sl, nl0, sl0)

plt.xlabel('bin lenght')
plt.ylabel(r'non local coeff')
plt.show()
plt.errorbar(xk, nlk, slk)

plt.xlabel('k')
plt.ylabel(r'non local coeff')
plt.show()

