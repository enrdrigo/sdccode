from modules import initialize
from modules import computenltt
import numpy as np
import os

root = './'
filename = 'dump1.1fs.lammpstrj'
fileinit = 'init.dat'

posox = float(input('position of the oxy:>\n'))
nkpoints = 500
ntrysnap = -1
if os.path.exists(root+filename):
    inputcompute = initialize.getinitialize(filename, root, posox, nkpoints, ntrysnap)
else:
    inputcompute = initialize.getinitialize(fileinit, root, posox, nkpoints, ntrysnap)

temp = float(input('temperature:>\n'))

natpermol = int(input('number of atoms per molecule:>\n'))

cp = float(input('specific heat in SI units:>\n'))

deltat = float(input('time step in metal units:>\n'))

tdump = int(input('dump interval:>\n'))


nltt, chi, corrk = computenltt.computenltt(inputcompute['root'], inputcompute['N'], inputcompute['size'],
                                           inputcompute['number of k'],  cp, deltat, tdump)

np.save(root+'nltt.npy', nltt)

xk = np.linspace(1, nkpoints, nkpoints-1) * 2 * np.pi / (inputcompute['size'])
with open(root+'nlttk.out', '+w') as f:
    for i in range(nkpoints-1):
        f.write('{}\t'.format(xk[i])+'{}\t'.format(np.real(np.mean(nltt[:, i, -1], axis=0)))+'{}\t'.format(chi[i+1])+'{}\n'.format(np.real(np.mean(np.sum(corrk[:,i, :int(corrk.shape[2]/2)+1], axis=1)))))

with open(root+'corren.out', 'w+') as f:
    for i in range(corrk.shape[2]):
        f.write('{}\t'.format(i)+'{}\t'.format(np.real(np.mean(corrk[:,0, i], axis=0)))+'{}\n'.format(np.real(np.mean(corrk[:,1, i], axis=0))))
