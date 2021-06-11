import numpy as np
import os
import time
from modules import initialize
from modules import dipole
from modules import computestatdc


# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZATION

start = time.time()
# INDICATES WHERE ARE THE DATA
root = '/Users/enricodrigo/Documents/LAMMPS/125_mol/04700/'
filename = 'dump1.05fs.lammpstrj'
# GETS THE NUMBER OF PARTICLES IN THE SIMULATION
Npart = initialize.getNpart(filename, root)
# IF NOT ALREADY DONE SAVE THE DATA IN A BINARY FORMAT SO THAT IN THE FUTURE READS JUST THE BINARY
if os.path.exists(root+'data'+'{}'.format(Npart)+'.npy'):
    filebin = 'data'+'{}'.format(Npart)+'.npy'
else:
    filebin = initialize.saveonbin(filename, root, Npart)  # JUST DO ONCE!!!! READ DATA FROM THE BINARY INSTEAD
# GETS THE DIMENSIONS OF THE SIMULATION BOX
Lato, Lmin = initialize.getBoxboundary(filename, root)
# GETS THE NUMBER OF SNAPSHOT OF THE SIMULATION. RESHAPE THE ARRAY OF THE DATA SO THAT WE HAVE A MATRIX WITH THE
# COORDINATES AND THE CHARGES FOR EACH SNAPSHOT
data_arrayy, nsnapshot = initialize.getDatainshape_andNsnap(filebin, root, Npart)
print('The data are stored in '+root+filename + '\n and written in bynary form in '+root+filebin)
print('The system has {}'.format(Npart)+' atoms in a box of side {:10.5f}'.format(Lato)+' Angstom')
print('In the calculation we are using {}'.format(nsnapshot)+' snapshots')
print('Initialization done in {:10.5f}'. format(time.time()-start)+'s')


# ----------------------------------------------------------------------------------------------------------------------
# CALCULATION OF THE DIPOLES

start1 = time.time()
# COMPUTES THE MATRIX OF THE MOLECULAR DIPOLES, THE CENTER OF MASS OF THE MOLECULE, THE ATOMIC CHARGES AND
# THE POSITION OF THE CHARGES (IN TIP4P/2005 THE OXY CHARGE IS IN A DIFFERENT POSITION THAN THE OXY ITSELF)
# POSO SETS THE DISTANCE BETWEEN THE OXY ATOM AND THE OXY CHARGE, IN THE TIP4P MODEL THE POSITIONS ARE DIFFERENT
posox=0.1250
dipmol, cdmol, chat, pos, enat, em, posatomic = dipole.computedipole(Npart, Lato, Lmin, nsnapshot, data_arrayy, posox)

print("Molecular dipoles, molecular positions, charges and charge positions for the trajectory computed in {:10.5f}".format(time.time()-start1)+'s')


# ----------------------------------------------------------------------------------------------------------------------
# CALCULATION OF THE STATIC DIELECTRIC CONSTANT

start2 = time.time()
nk = 120
# COMPUTES THE STATIC DIELECTRIC CONSTANT FOR NK VALUES OF THE G VECTOR IN THE (1,0,0) DIRECTION:
# 2\PI/LATO*(J,0,0), J=1,..NK
e0pol, e0ch = computestatdc.computestatdc(nk, dipmol, cdmol, chat, pos, Lato, 1)
e0pol, e0ch = computestatdc.computestatdc(nk, dipmol, cdmol, chat, pos, Lato, nsnapshot)
print('Static dielectric constant for {}'.format(nk)+' values of k computed in {:10.5f}'.format(time.time()-start2)+'s')


# ----------------------------------------------------------------------------------------------------------------------
# CALCULATION OF THE THERMOPOLARIZATION COEFFICIENT

start2 = time.time()
nk = 120
# COMPUTES THE THERMOPOLARIZATION COEFFICIENT FOR NK VALUES OF THE G VECTOR IN THE (1,0,0) DIRECTION:
# 2\PI/LATO*(J,0,0), J=1,..NK
tpc = computestatdc.thermopolcoeff(nk, chat, enat,em, pos, posatomic, Lato, nsnapshot)
print('Thermopolarization coefficient for {}'.format(nk)+' values of k computed in {:10.5f}'.format(time.time()-start2)+'s')


# ----------------------------------------------------------------------------------------------------------------------
# PRINT THE DIELECTRIC CONSTANT COMPUTED VIA THE POLARIZATION AND VIA THE CHARGES AS A FUNCTION OF G AND SAVE IN A FILE

file = '{}'.format(Npart)+'{}'.format(nk)+'dielconst.dat'
f = open(file, 'w+')
print("k\t"+'e0pol_xx\t'+'e0pol_yy\t'+'e0pol_zz\t'+'e0ch\t'+'thermopolarization coeff\n')
np.set_printoptions(precision=3)
f.write('#k (in units of 2pi/L(1,0,0))\t'+'e0pol_xx\t'+'e0pol_yy\t'+'e0pol_zz\t'+'e0ch\t'+'thermopolarization coeff\n')
for j in range(nk):
    print('{}\t'.format(j)+'{:10.3f}\t'.format(e0pol[j][0])+'{:10.3f}\t'.format(e0pol[j][1])+'{:10.3f}\t'.format(e0pol[j][2])+'{:10.3f}\t'.format(e0ch[j])+'{:.2e}'.format(tpc[j]))
    f.write('{}\t'.format(j)+'{:10.3f}\t'.format(e0pol[j][0])+'{:10.3f}\t'.format(e0pol[j][1])+'{:10.3f}\t'.format(e0pol[j][2])+'{:10.3f}\t'.format(e0ch[j])+'{:.2e}\t'.format(np.real(tpc[j]))+'{:.2e}\n'.format(np.imag(tpc[j])))
print('The static dielectric constants are saved in '+root+file)
print('The static dielectric constant and thermopolarization coefficient for {}'.format(nk)+' values of k computed in : {:10.5f}'.format(time.time()-start2)+'s')
f.close()


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE DIPOLES PAIR CORRELATION FUNCTION AT GMIN IN THE (G,0,0) DIRECTION AND SAVE IT IN A FILE
c = False
if c:
    G = 0
    start2 = time.time()
    file = '{}'.format(Npart)+'{}'.format(G)+'dippcfwstd.dat'
    print('The dipole pair correlation function for G=({}, 0, 0)'.format(G)+' is saved in '+root+file)
    rdipmol, rcdmol, tdipmol, tcdmol = computestatdc.reshape(cdmol, dipmol)
    nkk = 1
    print('r\t'+'c_m_x\t'+'c_m_y\t'+'c_m_z\t'+'std_c_m_x\t'+'std_c_m_y\t'+'std_c_m_z\t')
    gk, stdgk = computestatdc.dip_paircf(G, nkk, rdipmol, rcdmol, tdipmol, tcdmol, Lato, nsnapshot)
    print(time.time()-start2)
    start2=time.time()
    nkk = 100
    print('r\t'+'c_m_x\t'+'c_m_y\t'+'c_m_z\t'+'std_c_m_x\t'+'std_c_m_y\t'+'std_c_m_z\t')
    gk, stdgk = computestatdc.dip_paircf(G, nkk, rdipmol, rcdmol, tdipmol, tcdmol, Lato, nsnapshot)
    f = open(file, 'w+')
    for i in range(nk):
        f.write('{:10.5f}\t'.format((i*(Lato - 2)/2/nkk + 2)/0.529) + '{:10.5f}\t'.format(gk[i][0]) + '{:10.5f}\t'.format(gk[i][1]) + '{:10.5f}\t'.format(gk[i][2])+\
                '{:10.5f}\t'.format(stdgk[i][0]) + '{:10.5f}\t'.format(stdgk[i][1]) + '{:10.5f}\n'.format(stdgk[i][2]))
    print('The dipole pair correlation function for G=({}, 0, 0)'.format(G)+'  computed in : {:10.5f}'.format(time.time()-start2)+'s')
    print('Total elapsed time: {:10.5f}'.format(time.time()-start)+'s')
    f.close()
