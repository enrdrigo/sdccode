from modules import initialize
from modules import computenltt
import numpy as np
import os

root = './'
filename = 'dump1.1fs.lammpstrj'
fileinit = 'init.dat'

posox = 0.125
nkpoints = 40
ntrysnap = -1
if os.path.exists(root+filename):
    inputcompute = initialize.getinitialize(filename, root, posox, nkpoints, ntrysnap)
else:
    inputcompute = initialize.getinitialize(fileinit, root, posox, nkpoints, ntrysnap)

temp = float(input('temperature:>'))

nltt = computenltt.computenltt(inputcompute['root'], inputcompute['filename'], inputcompute['N'],
                               inputcompute['size'], inputcompute['position of the ox'],
                               inputcompute['number of k'], inputcompute['number of snapshots'], temp)

np.save(root+'nltt.npy', nltt)