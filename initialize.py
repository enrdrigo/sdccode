import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# READS THE DATA FROM THE LAMMPS OUTPUT AND SAVE IT IN A BINARY FORM. IT CAN TAKE VERY LONG.

def saveonbin(filename, root, Np):
    f = open(root + filename, 'r')
    f1 = f.readlines()
    d = []
    for line in f1:
        if len(line.split(' ')) != 8:
            continue
        dlist = [float(x.strip('\n')) for x in line.split(' ')]
        d.append(dlist)
    data_arrayy = np.array(d)
    filesavebin = 'data{}.npy'.format(Np)
    np.save(root + filesavebin, data_arrayy, allow_pickle=True)
    return filesavebin


# ----------------------------------------------------------------------------------------------------------------------
# GETS FROM THE LAMMPS OUTPUT THE NUMBER OF ATOMS IN THE SIMULATION

def getNpart(filename, root):
    f = open(root + filename, 'r')
    oldline = 'noline'
    flines = f.readlines()
    for line in flines:
        if oldline == 'ITEM: NUMBER OF ATOMS\n':
            Npart = int(line.split()[0])
            return Npart
        oldline = line


# ----------------------------------------------------------------------------------------------------------------------
# GETS FROM THE LAMMPS OUTPUT THE DIMENTION OF THE SIDE OF THE SIMULATION BOX

def getBoxboundary(filename, root):
    f = open(root + filename, 'r')
    oldline = 'noline'
    flines = f.readlines()
    for line in flines:
        if oldline == 'ITEM: BOX BOUNDS pp pp pp\n':
            (Linf, Lmax) = (float(line.split()[0]), float(line.split()[1]))
            return Lmax - Linf, Linf
        oldline = line


# ----------------------------------------------------------------------------------------------------------------------
# GETS THE NUMBER OF SNAPSHOT THAT WE ARE CONSIDERING AND PERFORMS A RESHAPE OF THE DATA ARRAY SO THAT WE HAVE FOR EACH
# SNAPSHOT A MATRIX WITH THE POSITION AND THE CHARGES OF THE MOLECULES

def getDatainshape_andNsnap(filename, root, Np):
    dati = np.load(root + filename, allow_pickle=True)
    nsnap = int(len(dati) / Np)
    data_arrayy = np.reshape(dati, (nsnap, Np, 8))
    return data_arrayy, nsnap
