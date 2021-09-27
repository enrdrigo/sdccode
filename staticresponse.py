from modules import initialize
from modules import compute
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

staticresponse = compute.computestaticresponse(inputcompute['root'], inputcompute['filename'], inputcompute['N'],
                                               inputcompute['size'], inputcompute['position of the ox'],
                                               inputcompute['number of k'], inputcompute['number of snapshots'], temp, natpermol)

