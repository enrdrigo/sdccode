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
    np.save(root + filebin, dati)

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
f = open('testconvergencenltt'+'{}'.format(Npart)+'.dat', 'w+')
timestep = float(input('timestep:\n'))
dumpstep = float(input('dumpstep:\n'))
dt = timestep*dumpstep
print('inizio calcolo non local thermal conductivity:')
rho = Npart/(6.022e23 * L ** 3 * 1.e-30)  # mol/m^3
cp = 18.0e-3  # Kcal/mol*k
fac = 4186*rho*cp  # J/k/m^3
nltt = []
sdd = []
taumax = int(nsnap/nsnap)+1

quota = 1

nltts = np.zeros(nk - 1)
corr = np.zeros((nk, int(nsnap/quota)), dtype=np.complex_)
result = np.zeros(2 * int(nsnap/quota), dtype=np.complex_)


for j in range(nk):
    result = signal.correlate((q[j, :int(nsnap/quota)]), q[j, :int(nsnap/quota)], mode='full', method='fft')
    v = [result[i] / (len(q[j, :int(nsnap/quota)]) - abs(i - (len(q[j, :int(nsnap/quota)])) + 1)) for i in range(len(result))]
    corr[j] = np.array(v[int(result.size / 2):])


with open('timecorrelationenen.dat', 'w+') as g:
    for i in range(len(corr[1])):
        g.write('{}\t'.format(i*dt/1000))
        g.write('{}\n'.format(np.real(corr[1, i]/corr[1, 0])))

for t in range(int(nsnap / quota) - 1):

    if t % int(nsnap / quota / 10) == 0:

        print('#{}'.format(int(t / (nsnap / quota) * 100) + 1) + '% ', end='', flush=True)

    ft = np.zeros(nk, dtype=np.complex_)
    for i in range(nk):

        ft[i] = np.sum(corr[i, :t+1]) * dt

    chi = np.var(q1[:, :], axis=1)

    for i in range(1, nk):

        nltts[i-1] = chi[i] / np.real(ft[i] * (2 * i * np.pi / L) ** 2) * fac * 1e-10 ** 2 / 1e-15  # m^2/s * J/k*m^3 = kJ/s /m

    f.write('{}\t'.format(t*dt/1000))
    f.write('{}\t'.format(nltts[0]))
    f.write('{}\t'.format(chi[1]))
    f.write('{}\n'.format(np.real(ft[1])))
print('')
print('Done')
f.close()



