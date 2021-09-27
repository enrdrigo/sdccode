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

cp = float(input('specific heat in metal units:>'))

deltat = float(input('time step in metal units:>'))

tdump = int(input('dump interval:>'))

nltt = computenltt.computenltt(inputcompute['root'], inputcompute['filename'], inputcompute['N'],
                               inputcompute['size'], inputcompute['position of the ox'],
                               inputcompute['number of k'], inputcompute['number of snapshots'], temp, natpermol, cp, deltat, tdump)

np.save(root+'nltt.npy', nltt)