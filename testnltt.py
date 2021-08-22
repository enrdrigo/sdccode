from modules import initialize
from modules import dipole
import numpy as np
from scipy import signal
from scipy.fft import fft
import os


root='../'
filename='dump1.1fs.lammpstrj'
L,Linf=initialize.getBoxboundary(filename, root)
Npart=375
print('lato cella', L)
f=open('file.out', '+w')
f.write('lato cella'+'{}\n'.format(L))
f.close()


if os.path.exists(root+filename+'{}'.format(Npart)+'.npz'):
    filebin = filename+'{}'.format(Npart)+'.npz'
else:
    filebin = initialize.saveonbin(filename, root, Npart)


nsnap = initialize.getNsnap(filebin, root, Npart)


g = open('file.out', '+w')
g.write('got nsnap\n')
g.close()


dip_mol, cdmol, ch_at, pos_ch, en_at, em, endip, pos_at=dipole.computedipole(Npart, L, Linf,nsnap, filebin, root, 0.125)


nk=20
nplot=4
nkpl=10
q=np.zeros((nk, nsnap), dtype=np.complex_)
q1=np.zeros((nk, nsnap), dtype=np.complex_)
for i in range(nk):
    q1[i]=np.sum((en_at[:,:])*np.exp(-1j*pos_at[:,:,0]*2*i*np.pi/L), axis=1)
qm=np.sum(q1, axis=1)/nsnap
q=np.transpose(np.transpose(q1)-qm)


corr=np.zeros((nk, nsnap), dtype=np.complex_)
result=np.zeros(2*nsnap, dtype=np.complex_)
for j in range(nk):
    result = 0
    result = signal.correlate((q[j,:]),q[j,:],  mode='full', method='fft')
    v = 0
    v = [result[i] /( len(q[j,:])-abs( i - (len(q[j,:])) +1 ) ) for i in range(len(result))]
    corr[j] = np.array(v[int(result.size/2):])


xx=np.linspace(0,nsnap-1,nsnap)
ft=np.zeros((nk, int(nsnap/4)), dtype=np.complex_)
for i in range(nk):
    ft[i]=fft(np.real(corr[i][:int(nsnap/4)]))


chi=np.var(q1, axis=1)
nltt=np.zeros(nk-1)
for i in range(1,nk):
    nltt[i-1]=chi[i]/(ft[i, 0]*(2*(i)*np.pi/L)**2)*1.e-5
    print(i, nltt[i-1], chi[i], ft[i, 0])


xk=np.linspace(1,nk, nk-1)
f=open('nltt.dat', 'w+')
for i in range(nk-1):
    f.write('{}\t'.format(xk[i]*2*np.pi/L*10)+'{}\n'.format(nltt[i]))
f.close()


tnltt=np.zeros(100)
for j in range(1,101):
    tauft=np.zeros((nk, int(j*nsnap/100)), dtype=np.complex_)
    tauft=fft(np.real(corr[1][:int(j*nsnap/100)]))
    tnltt[j-1]=chi[1]/(tauft[0]*(2*np.pi/L)**2)
xtau=np.linspace(0,nsnap, 100)
