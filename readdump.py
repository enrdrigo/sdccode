import numpy as np
import os
from modules import compute
from modules import initialize

root = './'
filename = 'dump1.1fs.lammpstrj'
fileinit = 'init.dat'

posox = float(input('position of the oxy:>\n'))
nkpoints = 10
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

compute.computekft(inputcompute['root'], inputcompute['filename'], inputcompute['N'],
                               inputcompute['size'], inputcompute['position of the ox'],
                               inputcompute['number of k'], inputcompute['number of snapshots'], natpermol)