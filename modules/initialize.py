import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# READS THE DATA FROM THE LAMMPS OUTPUT AND SAVE IT IN A BINARY FORM. IT CAN TAKE VERY LONG.

def saveonbin(filename, root, Np):
    f = open(root + filename, 'r')
    line = f.readline()
    d = []
    while (line != ''):
        if len(line.split(' ')) != 8:
            line = f.readline()
            continue
        dlist = [float(x.strip('\n')) for x in line.split(' ')]
        line = f.readline()
        d.append(dlist)
    f.close()
    data_arrayy = np.array(d)
    filesavebin = filename + '{}.npy'.format(Np)
    np.save(root + filesavebin, data_arrayy, allow_pickle=True)
    print('Done', data_arrayy, root + filesavebin)
    return filesavebin


# ----------------------------------------------------------------------------------------------------------------------
# GETS FROM THE LAMMPS OUTPUT THE NUMBER OF ATOMS IN THE SIMULATION

def getNpart(filename, root):
    with open(root + filename, 'r') as f:
        oldline = 'noline'
        for i in range(15):
            line = f.readline()
            if oldline == 'ITEM: NUMBER OF ATOMS\n':
                Npart = int(line.split()[0])
                f.close()
                return Npart
            oldline = line


# ----------------------------------------------------------------------------------------------------------------------
# GETS FROM THE LAMMPS OUTPUT THE DIMENTION OF THE SIDE OF THE SIMULATION BOX

def getBoxboundary(filename, root):
    f = open(root + filename, 'r')
    oldline = 'noline'
    for i in range(15):
        line = f.readline()
        if oldline == 'ITEM: BOX BOUNDS pp pp pp\n':
            (Linf, Lmax) = (float(line.split()[0]), float(line.split()[1]))
            f.close()
            return Lmax - Linf, Linf
        oldline = line


# ----------------------------------------------------------------------------------------------------------------------
# GETS THE NUMBER OF SNAPSHOT THAT WE ARE CONSIDERING AND PERFORMS A RESHAPE OF THE DATA ARRAY SO THAT WE HAVE FOR EACH
# SNAPSHOT A MATRIX WITH THE POSITION AND THE CHARGES OF THE MOLECULES

def getNsnap(filename, root, Np):
    dati = np.load(root + filename, allow_pickle=True)
    nsnap = int(len(dati) / Np)
    return nsnap
