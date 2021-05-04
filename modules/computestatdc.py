import numpy as np
#from numba import njit, prange, objmode

#-----------------------------------------------------------------------------------------------------------------------
#COMPUTES THE STATIC DIELECTRIC CONSTANT (SDC) AT A GIVEN PHYSICAL G VECTOR. HERE WE RESTRICT OURSELVES IN THE DIRECTION (1,0,0).
#THE SDC IS CALCULATED VIA THE AVERAGE VALUE OF THE MODULUS SQUARED OF THE FOURIER TRANFORM OF THE  POLARIZATION IN G OR VIA THE
# MODULUS SQUARED OF THE FOURIER TRANSORM OF THE CHARGE IN G DIVIDED BY THE MODULUS OF G.
#@njit(fastmath=True, parallel=True)
def computestatdc(nk, dipmol, cdmol,chat,pos, L, nsnap):
    e0pol=np.zeros((nk,3))
    e0ch=np.zeros(nk)
    for j in range(nk):
        e0pol[j][0]=1+(16.022**2)*np.sum(np.abs(np.sum(np.transpose(dipmol)[0]*\
                    np.exp(1j*np.transpose(cdmol)[0]*(j*2*np.pi/L)), axis=0))**2)/nsnap\
                    *1.0e5/((L)**3*1.38*300*8.854)
        e0pol[j][1]=1+(16.022**2)*np.sum(np.abs(np.sum(np.transpose(dipmol)[1]*\
                    np.exp(1j*np.transpose(cdmol)[0]*(j*2*np.pi/L)), axis=0))**2)/nsnap\
                    *1.0e5/((L)**3*1.38*300*8.854)
        e0pol[j][2]=1+(16.022**2)*np.sum(np.abs(np.sum(np.transpose(dipmol)[2]*\
                    np.exp(1j*np.transpose(cdmol)[0]*(j*2*np.pi/L)), axis=0))**2)/nsnap\
                    *1.0e5/((L)**3*1.38*300*8.854)
        e0ch[j]=1+(16.022**2)*np.mean(np.abs(np.sum(np.transpose(chat)*\
                np.exp(1j*np.transpose(pos)[0]*(j*2*np.pi/L)), axis=0))**2)\
                /((j*2*np.pi/L+ np.sqrt(3)*2*np.pi/L*1e-8)**2)*1.0e5/((L)**3*1.38*300*8.854)
    return e0pol, e0ch
#-----------------------------------------------------------------------------------------------------------------------