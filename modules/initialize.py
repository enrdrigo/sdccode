import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# READS THE DATA FROM THE LAMMPS OUTPUT AND SAVE IT IN A BINARY FORM. IT CAN TAKE VERY LONG.

def getdatafromfile(filename, root, Np):
    with  open(root + filename, 'r') as f:
        line = f.readline()
        d = []
        g = open('file.out', '+w')
        g.write('start translation in bin\n')
        g.close()
        while (line != ''):
            if len(line.split(' ')) != 8:
                line = f.readline()
                continue
            dlist = [float(x.strip('\n')) for x in line.split(' ')]
            line = f.readline()
            d.append(dlist)
        g = open('file.out', '+w')
        g.write('done translation in bin\n')
        g.close()

    return d


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
    with open(root + filename, 'r') as f:
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

def getNsnap(dati, Np):
    nsnap = int(len(dati) / Np)
    print(nsnap)
    return nsnap
