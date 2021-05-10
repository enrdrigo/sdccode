import numpy as np
from numba import njit
import time
# prange, objmode


# ----------------------------------------------------------------------------------------------------------------------
# COMPUTES THE STATIC DIELECTRIC CONSTANT (SDC) AT A GIVEN PHYSICAL G VECTOR. HERE WE RESTRICT OURSELVES IN THE
# DIRECTION (1,0,0). THE SDC IS CALCULATED VIA THE AVERAGE VALUE OF THE MODULUS SQUARED OF THE FOURIER TRANFORM OF THE
# POLARIZATION IN G OR VIA THE MODULUS SQUARED OF THE FOURIER TRANSORM OF THE CHARGE IN G DIVIDED BY THE MODULUS OF G.

@njit(fastmath=True, parallel=True)
def computestatdc(nk, dipmol, cdmol, chat, pos, L, nsnap):
    e0pol=np.zeros((nk,3))
    e0ch=np.zeros(nk)
    for j in range(nk):
        e0pol[j][0] = 1 + (16.022**2) * np.sum(np.abs(np.sum(np.transpose(dipmol)[0] *\
                    np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L))), axis=0))**2)/nsnap\
                    * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
        e0pol[j][1] = 1 + (16.022**2) * np.sum(np.abs(np.sum(np.transpose(dipmol)[1] *\
                    np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L))), axis=0))**2)/nsnap\
                    * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
        e0pol[j][2] = 1 + (16.022**2) * np.sum(np.abs(np.sum(np.transpose(dipmol)[2] *\
                    np.exp(1j * (np.transpose(cdmol)[0] * (j * 2 * np.pi / L))), axis=0))**2)/nsnap\
                    * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
        e0ch[j] = 1 + (16.022**2) * np.mean(np.abs(np.sum(np.transpose(chat) *\
                np.exp(1j * (np.transpose(pos)[0] * (j * 2 * np.pi / L))), axis=0))**2)\
                / ((j * 2 * np.pi / L + np.sqrt(3) * 2 * np.pi / L * 1e-8)**2) * 1.0e5 / (L**3 * 1.38 * 300 * 8.854)
    return e0pol, e0ch

def reshape(cdmol, dipmol):
    nmol=np.shape(dipmol)[1]
    nsnap=np.shape(dipmol)[0]
    # DIPMOL, CDMOL HAVE DIMENTION (NSNAP, NMOL, 3)
    rcdmol = np.transpose(cdmol, (1, 2, 0))
    rdipmol = np.transpose(dipmol, (1, 2, 0))
    tcdmol = np.zeros((3, nmol, nsnap))
    tcdmol = np.transpose(rcdmol, (1, 0, 2))
    tdipmol = np.zeros((3, nmol, nsnap))
    tdipmol = np.transpose(rdipmol, (1, 0, 2))
    rcdmol = np.reshape(rcdmol, (nmol, 1, 3, 1, nsnap))
    rdipmol = np.reshape(rdipmol, (nmol, 3, 1, nsnap))
    return rdipmol, rcdmol, tdipmol, tcdmol

# ----------------------------------------------------------------------------------------------------------------------
def dip_paircf(G, nk, rdipmol, rcdmol, tdipmol, tcdmol, L, nsnap):
    nmol = np.shape(rdipmol)[0]
    distcdm = np.zeros((nmol, 1, nmol, nsnap))
    diffcdm = np.zeros((nmol, 1, 3, nmol, nsnap))
    dipsq = np.zeros((nmol, 3, nmol, nsnap), dtype=np.complex_)
    gk = np.zeros((nk, 3))
    stdgk = np.zeros((nk, 3))
    sigma = 0.08
    cm = np.zeros((nmol, 3, nmol, nsnap), dtype=np.complex_)
    # RCDMOL, RDIPMOL HAVE SIZE (NMOL, 1, 3, 1, NSNAP), (NMOL, 3, 1, NSNAP)

    for i in range(nk):
        r = i * (L - 2) / 2 / nk + 2
        start=time.time()
        for s in range(nmol):

            diffcdm[s, :, :, :, :] = rcdmol[s, :, :, :, :]-tcdmol[:, :, :]

            distcdm[s, :, :, :] = np.sqrt(np.sum(diffcdm[s, :, :, :, :]**2, axis=1))

            dipsq[s, :, :, :] = rdipmol[s, :, :, :]*tdipmol[:, :, :] * np.exp(1j * diffcdm[s, :, 0, :, :] * (G * 2 * np.pi / L))

            dipsq[s, :, s, :] = 0

            cm[s, :, :, :] = dipsq[s, :, :, :]*np.exp(-(r-distcdm[s, :, :, :])**2/sigma**2)

        gk[i] = np.real(np.sum(np.sum(np.sum(cm, axis=0), axis=1)/nmol, axis=1)/nsnap/r**2/2*np.pi/sigma**2)
        stdgk[i] = np.sqrt(np.real(np.var(np.sum(np.sum(cm, axis=0), axis=1)/nmol, axis=1)/nsnap)/r**2/2*np.pi/sigma**2)
        print('{:10.5f}\t'.format((i*(L - 2)/2/nk + 2)/0.529) + '{:10.5f}\t'.format(gk[i][0]) + '{:10.5f}\t'.format(gk[i][1]) + '{:10.5f}\t'.format(gk[i][2])+\
            '{:10.5f}\t'.format(stdgk[i][0]) + '{:10.5f}\t'.format(stdgk[i][1]) + '{:10.5f}\t'.format(stdgk[i][2]), time.time()-start)

    return gk, stdgk

# cm = np.zeros((3, nmol, nsnap), dtype=np.complex_)
# for s in range(nk):
#     r = (s+1)*L/nk
#     for i in range(nmol):
#         diffcdm= np.transpose(cdmol, [1, 0, 2]) - np.transpose(cdmol, [1, 0, 2])[i]
#         distcdm = np.sqrt(np.sum(diffcdm**2, axis=2))
#         dipsq = np.transpose(np.transpose(np.transpose(dipmol, [1, 0, 2]) * np.transpose(dipmol, [1, 0, 2])[i]) \
#                 * np.exp(1j * np.transpose(diffcdm, [2, 1, 0])[0] * (1 * 2 * np.pi / L)))
#         # dipsq = np.transpose(dipmol, [1, 0, 2])*np.transpose(dipmol, [1, 0, 2])[i]\
#         #         * np.exp(1j*np.transpose(diffcdm, [2, 0, 1])[0]*(1*2*np.pi/L))
#         dipsq[i, :] = 0
#         cm[:, i, :] = np.sum(np.transpose(dipsq, [2, 0, 1])*np.exp(-(r-distcdm)**2/sigma**2), axis=1)
#     gk[s] = np.real(np.sum(np.sum(cm, axis=1)/nmol, axis=1)/nsnap/r**2/2*np.pi/sigma**2)
#     print(r/0.529, gk[s])

#    for i in range(nmol):
#        # 1=NMOL, 0=NSNAP, 2=3
#        diffcdm= np.transpose(cdmol, [1, 0, 2]) - np.transpose(cdmol, [1, 0, 2])[i]
#        distcdm = np.sqrt(np.sum(diffcdm**2, axis=2))
#        # 1=NMOL, 0=NSNAP, 2=3
#        dipsq = np.transpose(np.transpose(np.transpose(dipmol, [1, 0, 2]) * np.transpose(dipmol, [1, 0, 2])[i]) \
#                * np.exp(1j * np.transpose(diffcdm, [2, 1, 0])[0] * (0 * 2 * np.pi / L))) # 2=3, 1=NSNAP, 0=NMOL
#        dipsq[i] = 0
#        # 2=3, 1=NMOL, 1=NSNAP
#        cm[:, :, i, :] = np.sum(np.transpose(dipsq, [2, 0, 1])*np.exp(-(r-distcdm)**2/sigma**2), axis=2)
#    gk = np.real(np.sum(np.sum(cm, axis=2)/nmol, axis=2)/nsnap/rr**2/2*np.pi/sigma**2)



