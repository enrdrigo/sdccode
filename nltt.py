from modules import initialize
from modules import computenltt
import numpy as np
import os

root = './'
filename = 'dump1.1fs.lammpstrj'
fileinit = 'init.dat'

posox = float(input('position of the oxy:>'))
nkpoints = 40
ntrysnap = -1
if os.path.exists(root+filename):
    inputcompute = initialize.getinitialize(filename, root, posox, nkpoints, ntrysnap)
else:
    inputcompute = initialize.getinitialize(fileinit, root, posox, nkpoints, ntrysnap)

temp = float(input('temperature:>'))

natpermol = int(input('number of atoms per molecule:>'))

cp = float(input('specific heat in SI units:>'))

deltat = float(input('time step in metal units:>'))

tdump = int(input('dump interval:>'))

nltt, chi, corrk = computenltt.computenltt(inputcompute['root'], inputcompute['filename'], inputcompute['N'],
                               inputcompute['size'], inputcompute['position of the ox'],
                               inputcompute['number of k'], inputcompute['number of snapshots'], temp, natpermol, cp, deltat, tdump)

np.save(root+'nltt.npy', nltt)

xk = np.linspace(1, nkpoints, nkpoints-1) * 2 * np.pi / (inputcompute['size'])
with open(root+'nlttk.out', '+w') as f:
    for i in range(nkpoints-1):
        f.write('{}\t'.format(xk[i])+'{}\t'.format(np.real(np.mean(nltt[:, i, -1], axis=0)))+'{}\t'.format(chi[i+1])+'{}\n'.format(np.real(np.mean(corrk[:,i]))))