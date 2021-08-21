import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE POSITION OF THE OXY AND OF THE TWO HYDROGENS AT GIVEN SNAPSHOT. IT ALSO GETS THE POSITION OF THE
# FOURTH PARTICLE IN THE TIP4P/2005 MODEL OF WATER WHERE THERE IS THE CHARGE OF THE OXY (SEE TIP4P/2005 MODEL OF WATER).

def computeposmol(GG, Np, L, Linf, nsnap, data_array, posox):
    nmol = int(Np / 3)
    datamol = np.zeros((8, nmol, 3))
    datamol = data_array.reshape((8, nmol, 3))

    #
    posO = np.zeros((3, nmol))
    posH1 = np.zeros((3, nmol))
    posH2 = np.zeros((3, nmol))
    posO[0] = np.transpose(datamol[2])[0]
    posO[1] = np.transpose(datamol[3])[0]
    posO[2] = np.transpose(datamol[4])[0]
    posH1[0] = np.transpose(datamol[2])[1]
    posH1[1] = np.transpose(datamol[3])[1]
    posH1[2] = np.transpose(datamol[4])[1]
    posH2[0] = np.transpose(datamol[2])[2]
    posH2[1] = np.transpose(datamol[3])[2]
    posH2[2] = np.transpose(datamol[4])[2]
    #

    #
    bisdir = np.zeros((3, nmol))
    bisdir = 2 * posO - posH1 - posH2
    #

    #
    poschO = np.zeros((3, nmol))
    poschO = posO - posox * bisdir / np.sqrt(bisdir[0] ** 2 + bisdir[1] ** 2 + bisdir[2] ** 2)
    #

    return poschO, posO, posH1, posH2


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE MOLECULAR DIPOLES AND THE CENTER OF MASS OF THE MOLECULES AT GIVEN SNAPSHOT. COMPITING THE MOLECULAR
# DIPOLE WE MUST REMEMBER THAT THE OXY CHARGE IS NOT LOCATED IN THE OXY POSITION (SEE TIP4P/2005 MODEL OF WATER).

def computemol(GG, Np, L, Linf, nsnap, data_array, poschO, posO, posH1, posH2):
    nmol = int(Np / 3)
    datamol = np.zeros((8, nmol, 3))
    datamol = data_array.reshape((8, nmol, 3))

    #
    Gmol = np.ones((3, nmol))

    Gmol[0] = GG[0] * np.ones(nmol)
    Gmol[1] = GG[1] * np.ones(nmol)
    Gmol[2] = GG[2] * np.ones(nmol)
    #

    #
    chO = np.zeros(nmol)
    chH1 = np.zeros(nmol)
    chH2 = np.zeros(nmol)
    chO = np.transpose(datamol[5])[0]
    chH1 = np.transpose(data_array[5])[1]
    chH2 = np.transpose(data_array[5])[2]
    #

    #
    cdmmol = np.zeros((3, nmol))
    cdmmol = (posO * 15.9994 + (posH1 + posH2) * 1.008) / (15.9994 + 2 * 1.008)
    #

    #
    pos_mch = np.zeros((3, nmol))
    pos_mch = poschO * chO + posH1 * chH1 + posH2 * chH2
    #

    #
    dip_mol0 = np.zeros((3, nmol), dtype=np.complex_)
    dip_mol0 = pos_mch * np.exp(1j * np.sum(cdmmol * Gmol, axis=0))
    #

    return np.transpose(dip_mol0), np.transpose(cdmmol)


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE CHARGE AND ATOMIC POSITION ARRAYS OF THE ATOMS AT A GIVEN SNAPSHOT. THE OXY POSITION IS SHIFTED ACCORDING
# TO THE TIP4P/2005 MODEL OF WATER.

def computeat(GG, Np, L, Linf, nsnap, data_array, poschO, posH1, posH2):
    nmol = int(Np / 3)

    #
    Gat = np.ones((3, Np))

    Gat[0] = GG[0] * np.ones(Np)
    Gat[1] = GG[1] * np.ones(Np)
    Gat[2] = GG[2] * np.ones(Np)
    #

    #
    chat = np.zeros(Np)
    chat = data_array[5]
    #

    #
    posm = np.zeros((3, nmol, 3))
    posm[0] = np.transpose(poschO)
    posm[1] = np.transpose(posH1)
    posm[2] = np.transpose(posH2)
    #

    #
    pos_at = np.zeros((3, Np))
    test = posm.transpose()
    pos_at = test.reshape((3, Np))
    #

    # THIS IS IN FACT THE CHARGE TIMES A PHASE WHERE GAT = 2 * np.pi * np.array((1e-8, 1e-8, 1e-8)) / L. I DO THIS
    # IN ORDER TO COMPUTE PROPERLY THE STATIC DIELECTRIC CONSTANT VIA THE FOURIER TRANFORM OR THE CHARGE OVER
    # THE MODULUS OF G,  AT G \APPROX 0
    ch_at = np.zeros(Np, dtype=np.complex_)
    ch_at = chat * np.exp(1j * np.sum(pos_at * Gat, axis=0))
    #

    #
    dip_k_rm_at = np.zeros((3, Np))
    dip_k_rm_at = pos_at * chat * np.exp(1j * np.sum(pos_at * Gat, axis=0))
    #

    return ch_at, np.transpose(pos_at)


def computeaten(GG, Np, L, Linf, nsnap, data_array, poschO, posH1, posH2):
    nmol = int(Np / 3)
    #
    datamol = np.zeros((8, nmol, 3))
    datamol = data_array.reshape((8, nmol, 3))
    #
    en0 = np.zeros(nmol)
    enH1 = np.zeros(nmol)
    enH2 = np.zeros(nmol)
    enO = np.transpose(datamol[6])[0] + np.transpose(datamol[7])[0]
    enH1 = np.transpose(datamol[6])[1] + np.transpose(datamol[7])[1]
    enH2 = np.transpose(datamol[6])[2] + np.transpose(datamol[7])[2]
    #
    Gat = np.ones((3, Np))

    Gat[0] = GG[0] * np.ones(Np)
    Gat[1] = GG[1] * np.ones(Np)
    Gat[2] = GG[2] * np.ones(Np)
    #

    #
    enat = np.zeros(Np)
    enat = data_array[6] + data_array[7]
    #

    #
    pos_at = np.zeros((3, Np))
    pos_at[0] = data_array[2]
    pos_at[1] = data_array[3]
    pos_at[2] = data_array[4]
    #

    #
    en_at = np.zeros(Np, dtype=np.complex_)
    en_at = enat * np.exp(-1j * np.sum(pos_at * Gat, axis=0))
    #

    endip = np.zeros((3, nmol))
    endip = poschO*(enO-np.sum(enat)/Np) + posH1*(enH1-np.sum(enat)/Np) + posH2*(enH2-np.sum(enat)/Np)

    return en_at, np.transpose(pos_at), np.sum(enat), np.transpose(endip)


# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZES THE ARRAY NEEDED IN THE ROUTINE staticdc SO THAT IT IS MORE EASILY READABLE.

def initialize(nsnap, Np):
    nmol = int(Np / 3)
    cdmol = np.zeros((nsnap, nmol, 3))
    pos_at = np.zeros((nsnap, Np, 3))
    ch_at = np.zeros((nsnap, Np), dtype=np.complex_)
    dip_at = np.zeros((nsnap, Np, 3), dtype=np.complex_)
    dip_mol = np.zeros((nsnap, nmol, 3), dtype=np.complex_)
    endip = np.zeros((nsnap, nmol, 3))
    poschO = np.zeros((3, nmol))
    posO = np.zeros((3, nmol))
    posH1 = np.zeros((3, nmol))
    posH2 = np.zeros((3, nmol))
    posatomic= np.zeros((nsnap, Np, 3))
    em = np.zeros(nsnap)
    en_at = np.zeros((nsnap, Np), dtype=np.complex_)
    return cdmol, pos_at, ch_at, dip_at, dip_mol, en_at, em, endip, poschO, posO, posH1, posH2, posatomic


# ----------------------------------------------------------------------------------------------------------------------
# CASTS THE MOLECULAR DIPOLES, MOLECULAR CENTERS OF MASS, ATOMIC CHARGES AND ATOMIC POSITIONS IN NP.ARRAYS.

def computedipole(Np, L, Linf, nsnap, filename, root, posox):
    dati = np.load(root + filename, allow_pickle=True)
    data_array = np.reshape(dati, (nsnap, Np, 8))
    nmol = int(Np / 3)
    cdmol, pos_at, ch_at, dip_at, dip_mol, en_at, em, endip, poschO, posO, posH1, posH2, posatomic = initialize(nsnap, Np)
    G = np.zeros(3)
    G = 2 * np.pi * np.array((0, 0, 0)) / L
    for s in range(nsnap):
        poschO, posO, posH1, posH2 = computeposmol(G, Np, L, Linf, nsnap, data_array[s].transpose(), posox)

        dip_mol[s], cdmol[s] = computemol(G, Np, L, Linf, nsnap, data_array[s].transpose(), poschO, posO, posH1, posH2)

        ch_at[s], pos_at[s] = computeat(G, Np, L, Linf, nsnap, data_array[s].transpose(), poschO, posH1, posH2)

        en_at[s], posatomic[s], em[s], endip[s] = computeaten(G, Np, L, Linf, nsnap, data_array[s].transpose(), poschO, posH1, posH2)
    return dip_mol, cdmol, ch_at, pos_at, en_at, em, endip, posatomic

