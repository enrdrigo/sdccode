import numpy as np
import pickle as pk
from modules import compute
from numba import njit
import time
import h5py
import os


def read_dump(root, filename, Np, ntry):

    with open(root + filename, 'r') as f:

        if os.path.exists(root + 'dump.h5'):
            with h5py.File('dump.h5', 'r') as dump:
                snap = list(dump.keys())

            lenght = len(snap)
            print('THE LOADING WAS STOPPED AT THE SNAPSHOT: ', lenght)

        else:
            dump = h5py.File('dump.h5', 'a')
            lenght = 0

        dump = h5py.File('dump.h5', 'a')
        d = []
        start = time.time()

        for index, line in enumerate(f):

            if index < lenght * (Np + 9):
                continue

            linesplit = line.split(' ')

            if len(linesplit) != 8 and len(linesplit) != 9:
                continue

            dlist = [float(linesplit[i]) for i in range(8)]
            d.append(dlist)

            if (index + 1) % (Np + 9) == 0:

                if len(d) == 0:
                    print('END READ FILE')
                    print('got ' + str((index + 1) // (Np + 9)) + ' snapshot')
                    dump.close()
                    return

                elif len(d) != Np:

                    print(len(d))
                    print('STOP: THE SNAPSHOT ' + str((index + 1) // (Np + 9)) + ' DOES NOT HAVE ALL THE PARTICLES')
                    print('got ' + '' + ' snapshot')
                    dump.close()
                    return

                datisnap = np.array(d)
                d = []
                dump.create_dataset(str((index + 1) // (Np + 9)), data=datisnap)
                # print(index, (index + 1) / (Np + 9))

                if (index + 1) // (Np + 9) + 3 == ntry * 3:
                    print('number of total snapshots is', (index + 1) // (Np + 9) / 3)
                    print('done')
                    print('END READ. NO MORE DATA TO LOAD. SEE NTRY')
                    dump.close()
                    return

                if (index + 1) // (Np + 9) + 3 == ntry * 3:
                    print('number of total snapshots is', (index + 1) // (Np + 9))
                    print('done')
                    print('elapsed time: ', time.time() - start)
                    print('END READ NTRY')
                    dump.close()
                    return

        print('number of total snapshots is', (index + 1) // (Np + 9))
        print('done')
        print('elapsed time: ', time.time() - start)
        print('END READ FILE GOOD')
        dump.close()
        return


def computekftnumba(root, Np, L, posox, nk, ntry, natpermol):
    start0 = time.time()
    enk = []
    dipenkx = []
    dipenky = []
    chk = []
    dipkx = []
    dipky = []
    ifprint = False
    with open(root + 'output.out', 'a') as g:
        print('start the computation of the fourier transform of the densities')
        g.write('start the computation of the fourier transform of the densities\n')
    with h5py.File(root + 'dump.h5', 'r') as dump:
        print('tempo di apertira', time.time() - start0)
        snap = list(dump.keys())

        if os.path.exists(root + 'chk.pkl'):
            with open(root + 'enk.pkl', 'rb') as g:
                enk = pk.load(g)
            with open(root + 'dipenkx.pkl', 'rb') as g:
                dipenkx = pk.load(g)
            with open(root + 'dipenky.pkl', 'rb') as g:
                dipenky = pk.load(g)
            with open(root + 'chk.pkl', 'rb') as g:
                chk = pk.load(g)
            with open(root + 'dipkx.pkl', 'rb') as g:
                dipkx = pk.load(g)
            with open(root + 'dipky.pkl', 'rb') as g:
                dipky = pk.load(g)

            lenght = int(len(chk) / 3)

            if len(snap) != lenght:
                pass

            print('THE LOADING WAS STOPPED AT THE SNAPSHOT: ', lenght)

        else:
            lenght = 0

        for i in range(lenght + 1, len(snap) + 1):

            start1 = time.time()

            if ifprint:
                print(len(chk) / 3)

            datisnap = dump[str(i)][()]

            if ifprint:
                print('tempo ricerca nel dizionario', time.time() - start1)

            start2 = time.time()

            poschO, pos = compute.computeposmol(Np, datisnap.transpose(), posox, natpermol)

            dip_mol, cdmol = compute.computemol(Np, datisnap.transpose(), poschO, pos)

            ch_at, pos_at = compute.computeat(Np, datisnap.transpose(), poschO, pos)

            en_at, posatomic, em, endip = compute.computeaten(Np, datisnap.transpose(), pos)

            emp = em / Np * np.ones(Np)

            if ifprint:
                print('tempo calcolo funzioni', time.time() - start2)

            start3 = time.time()

            enklist, dipenkxlist, dipenkylist, chklist, dipkxlist, dipkylist \
                = numbacomputekft((en_at[:] - emp[:]), (endip[:, 0]), (endip[:, 1]), (ch_at[:]), (dip_mol[:, 0]),
                                  (dip_mol[:, 1]), \
                                  posatomic[:, 0], cdmol[:, 0], cdmol[:, 0], pos_at[:, 0], cdmol[:, 0], cdmol[:, 0], L,
                                  nk)

            enk.append(enklist)

            dipenkx.append(dipenkxlist)

            dipenky.append(dipenkylist)

            chk.append(chklist)

            dipkx.append(dipkxlist)

            dipky.append(dipkylist)

            enklist, dipenkxlist, dipenkylist, chklist, dipkxlist, dipkylist \
                = numbacomputekft((en_at[:] - emp[:]), (endip[:, 1]), (endip[:, 2]), (ch_at[:]), (dip_mol[:, 1]),
                                  (dip_mol[:, 2]), \
                                  posatomic[:, 1], cdmol[:, 1], cdmol[:, 0], pos_at[:, 1], cdmol[:, 1], cdmol[:, 0], L,
                                  nk)

            enk.append(enklist)

            dipenkx.append(dipenkxlist)

            dipenky.append(dipenkylist)

            chk.append(chklist)

            dipkx.append(dipkxlist)

            dipky.append(dipkylist)

            enklist, dipenkxlist, dipenkylist, chklist, dipkxlist, dipkylist \
                = numbacomputekft((en_at[:] - emp[:]), (endip[:, 2]), (endip[:, 2]), (ch_at[:]), (dip_mol[:, 2]),
                                  (dip_mol[:, 2]), \
                                  posatomic[:, 2], cdmol[:, 2], cdmol[:, 1], pos_at[:, 2], cdmol[:, 2], cdmol[:, 1], L,
                                  nk)

            enk.append(enklist)

            dipenkx.append(dipenkxlist)

            dipenky.append(dipenkylist)

            chk.append(chklist)

            dipkx.append(dipkxlist)

            dipky.append(dipkylist)

            if ifprint:
                print('tempo calcolo ftk', time.time() - start3)

            if int(len(chk) / 3 + 1) % int(len(snap)/10) == 0:
                print('got ' + str(len(chk) / 3) + ' snapshot' + '({}%)'.format(int(len(chk) / 3 + 1)*100//len(snap)+1))
                print('average elapsed time per snapshot', (time.time() - start0) / (1 + len(chk) / 3))

            if int(len(chk) / 3 + 1) % int(len(snap)/4+1) == 0:
                with open(root + 'output.out', 'a') as z:
                    with open(root + 'enk.pkl', 'wb+') as g:
                        pk.dump(enk, g)
                    with open(root + 'dipenkx.pkl', 'wb+') as g:
                        pk.dump(dipenkx, g)
                    with open(root + 'dipenky.pkl', 'wb+') as g:
                        pk.dump(dipenky, g)
                    with open(root + 'chk.pkl', 'wb+') as g:
                        pk.dump(chk, g)
                    with open(root + 'dipkx.pkl', 'wb+') as g:
                        pk.dump(dipkx, g)
                    with open(root + 'dipky.pkl', 'wb+') as g:
                        pk.dump(dipky, g)

                    print('got ' + str(len(chk) / 3) + ' snapshot')
                    print('average elapsed time per snapshot', (time.time() - start0) / (1 + len(chk) / 3))
                    z.write('got ' + str(len(chk) / 3) + ' snapshot\n')
                    z.write('average elapsed time per snapshot' + '{}\n'.format(
                        (time.time() - start0) / (1 + len(chk) / 3)))

            if len(chk) + 3 == ntry * 3:
                with open(root + 'enk.pkl', 'wb+') as g:
                    pk.dump(enk, g)
                with open(root + 'dipenkx.pkl', 'wb+') as g:
                    pk.dump(dipenkx, g)
                with open(root + 'dipenky.pkl', 'wb+') as g:
                    pk.dump(dipenky, g)
                with open(root + 'chk.pkl', 'wb+') as g:
                    pk.dump(chk, g)
                with open(root + 'dipkx.pkl', 'wb+') as g:
                    pk.dump(dipkx, g)
                with open(root + 'dipky.pkl', 'wb+') as g:
                    pk.dump(dipky, g)
                with open(root + 'output.out', 'a') as g:
                    print('number of total snapshots is', len(chk) / 3)
                    print('done')
                    print('elapsed time: ', time.time() - start0)
                    g.write('number of total snapshots is' + '{}\n'.format(len(chk) / 3))
                    g.write('done')
                print('END COMPUTE NTRY')
                return

        with open(root + 'output.out', 'a') as g:
            print('number of total snapshots is', len(chk) / 3)
            print('done')
            print('elapsed time: ', time.time() - start0)
            g.write('number of total snapshots is' + '{}\n'.format(len(chk) / 3))
            g.write('done')

        with open(root + 'enk.pkl', 'wb+') as g:
            pk.dump(enk, g)
        with open(root + 'dipenkx.pkl', 'wb+') as g:
            pk.dump(dipenkx, g)
        with open(root + 'dipenky.pkl', 'wb+') as g:
            pk.dump(dipenky, g)
        with open(root + 'chk.pkl', 'wb+') as g:
            pk.dump(chk, g)
        with open(root + 'dipkx.pkl', 'wb+') as g:
            pk.dump(dipkx, g)
        with open(root + 'dipky.pkl', 'wb+') as g:
            pk.dump(dipky, g)
        print('END COMPUTE GOOD')
        return

pa = bool(int(input('do you want multithreading? (0=False, 1=True)')))
@njit(fastmath=True, parallel=pa)
def numbacomputekft(f1, f2, f3, f4, f5, f6, x1, x2, x3, x4, x5, x6, L, nk):
    fk1 = [np.sum(f1 * np.exp(1j * x1 * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L)) for i in range(nk)]
    fk2 = [np.sum(f2 * np.exp(1j * x2 * 2 * -i * np.pi / L)) for i in range(nk)]
    fk3 = [np.sum(f3 * np.exp(1j * x3 * 2 * -i * np.pi / L)) for i in range(nk)]
    fk4 = [np.sum(f4 * np.exp(1j * x4 * 2 * -(i + np.sqrt(3.) * 1.0e-5) * np.pi / L)) for i in range(nk)]
    fk5 = [np.sum(f5 * np.exp(1j * x5 * 2 * -i * np.pi / L)) for i in range(nk)]
    fk6 = [np.sum(f6 * np.exp(1j * x6 * 2 * -i * np.pi / L)) for i in range(nk)]
    return fk1, fk2, fk3, fk4, fk5, fk6