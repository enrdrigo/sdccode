import numpy as np
from numpy.linalg import eig
from modules import cubicharmonics


def generatesorteddata(data, nk):
    G = cubicharmonics.Gvecgenerateall(nk)

    gmod = np.linalg.norm(G[1:], axis=1)

    dicdata = {}
    dicg = {}
    sdata = np.zeros((3, len(data[0])))
    nr = np.random.random(len(data[0])) * 1.0e-6
    for i in range(len(data[0])):
        dicdata[gmod[i] + nr[i]] = [data[1][i], data[2][i]]
        dicg[gmod[i] + nr[i]] = [G[i + 1][0], G[i + 1][1], G[i + 1][2]]
    data0sort = np.sort(gmod[:] + nr)

    sdata = np.zeros((2, len(data[0])))
    grid = np.zeros((len(data[0]), 3))
    for i in range(len(data[0])):
        grid[i][0] = dicg[data0sort[i]][0]
        grid[i][1] = dicg[data0sort[i]][1]
        grid[i][2] = dicg[data0sort[i]][2]
        sdata[0][i] = dicdata[data0sort[i]][0]
        sdata[1][i] = dicdata[data0sort[i]][1]
        dicdata[data0sort[i]][0]
    print(grid[0], grid[1], grid[2])
    return sdata, grid


def bayesianpol(grid, sdata, M, N, alpha, x_infer, ifprint=False):
    # grid e' la griglia di punti k.
    # data sono i valori calcolati nella simualzione con la std dev dei dati.
    # M e' il grado massimo dei polinomi che considero.
    # N e' il numero di dati nel fit.
    # alpha e' il parametro di regolarizzazione.
    # x_infer sono i punti k dove voglio inferire il risultato.
    sigma_noise = sdata[1][:N]
    x = grid[:N, :].T * 0.13484487571168569
    if ifprint: print(x.shape)
    y_noise = sdata[0][:N]
    betha = (1 / sigma_noise) ** 2

    if ifprint: print(cubicharmonics.computecubicar(2, x.T).shape)

    contanumpol = 1
    Phi = [np.ones((x.shape[1])) * np.sqrt(betha)]
    for i in range(1, M):
        s = cubicharmonics.computecubicar(i, x.T)
        contanumpol += s.shape[1]

        if ifprint: print(contanumpol, i, Phi[0].shape, s.shape)

        for j in range(s.shape[1]):
            Phi = np.append(Phi, [s[:, j] * np.sqrt(betha)], axis=0)

            if ifprint: print(contanumpol, 'in', Phi.shape)

        Phi = np.array(Phi)

        if ifprint: print(contanumpol, 'out', Phi.shape)
    if M == 1: Phi = np.array(Phi)

    SN_inv = alpha * np.identity(contanumpol) + np.dot(Phi, Phi.T)
    SN = np.linalg.inv(SN_inv)
    mN = np.dot(np.dot(SN, Phi), y_noise[:] * np.sqrt(betha))

    if ifprint: print(mN)
    if ifprint: print(cubicharmonics.computecubicar(2, x.T).shape)

    Phi_infer = [np.ones((x.shape[1]))]
    for i in range(1, M):
        s = cubicharmonics.computecubicar(i, x.T)

        if ifprint: print(i, Phi_infer[0].shape, s.shape)

        for j in range(s.shape[1]):
            Phi_infer = np.append(Phi_infer, [s[:, j]], axis=0)

            if ifprint: print('in', Phi_infer.shape)

        Phi_infer = np.array(Phi_infer)

        if ifprint: print('out', Phi_infer.shape)
    if M == 1: Phi_infer = np.array(Phi_infer)

    y_infer = np.dot(mN, Phi_infer)

    sy_infer = np.diag(np.dot(Phi_infer.T, np.dot(SN, Phi_infer)))

    if ifprint: print(y_infer[0], mN[0], sdata[0][0])

    return mN, SN, y_infer, sy_infer


def bestfit(grid, sdata, N, x_infer, ifprintcubic=False, ifprintbestfit=False):
    # grid e' la griglia di punti k.
    # sdata sono i valori calcolati nella simualzione con la std dev dei dati.
    # N e' il numero di dati nel fit.
    # x_infer sono i punti k dove voglio inferire il risultato.

    M_tot = 20
    # M_tot e' il numero massimo del grado del polinomio che considero
    log_evidence_vP = []
    alpha_vP = []
    betha_vP = np.zeros((M_tot))
    g_vP = np.zeros((M_tot))
    x = grid[:N, :].T * 0.13484487571168569
    y_noise = sdata[0][:N]
    Mv_list = []
    sigma_noise = sdata[1][:N]
    betha0 = (1 / sigma_noise) ** 2
    for M_v in range(1, M_tot):  # number of parameters

        # calcolo il set di funzioni di base (armoniche cubiche) associate a questo grado M_v
        contanumpol = 1
        Phi_vP = [np.ones((x.shape[1])) * np.sqrt(betha0)]
        for i in range(1, M_v):

            s = cubicharmonics.computecubicar(i, x.T)
            contanumpol += s.shape[1]

            if ifprintcubic: print('numero di polinomi fino al grado', 2 * i, ':', contanumpol, \
                                   'numero di armoniche cubiche:', s.shape[1] + 1)

            for j in range(s.shape[1]):
                Phi_vP = np.append(Phi_vP, [s[:, j] * np.sqrt(betha0)], axis=0)

            Phi_vP = np.array(Phi_vP)

        # contanumpol e' il numero di armoniche cubiche associate a M_v
        if M_v == 1: Phi_vP = np.array(Phi_vP)

        # calcolo gli autovalori di Phi_vP, servono per la stima di alpha ottimale
        li_vP, ei_vP = eig(np.dot(Phi_vP, Phi_vP.T))

        alpha0 = 1000
        delta_alphaP = 1
        alphaP = alpha0
        conta = 0

        # inizio il ciclo self-consistente per ottenere il valore migliore di alpha
        while abs(delta_alphaP / (alphaP + 0.1)) > 1e-10 and conta < 1.0e3:
            conta += 1

            SN_vP = np.linalg.inv(alphaP * np.identity(contanumpol) + np.dot(Phi_vP, Phi_vP.T))
            mN_vP = np.dot(np.dot(SN_vP, (Phi_vP)), y_noise * np.sqrt(betha0))
            g_vP = np.sum(li_vP.real / (alphaP + li_vP.real))
            alpha1P = g_vP / (np.dot(mN_vP.T, mN_vP))
            delta_alphaP = alpha1P - alphaP
            alphaP = alpha1P

        if (abs(delta_alphaP / (alphaP + 0.1)) > 1e-10):
            if ifprintbestfit: print('no convergence', N, x[-1], conta, delta_alphaP, alphaP, M_v)

        Mv_list.append(M_v)
        alpha_vP.append(alphaP)

        # mi preparo a calcolare la funzione di evidence per il valore ottimale di alpha
        A_vP = np.linalg.inv(SN_vP)
        E_mNs_vP = 1 / 2 * (y_noise * np.sqrt(betha0) - np.dot(Phi_vP.T, mN_vP.T)) ** 2
        E_mN_vP = E_mNs_vP.sum()
        log_evidence_vP.append(M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(betha0)) - \
                               E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        print(np.linalg.det(np.dot(Phi_vP, Phi_vP.T)))
        if ifprintbestfit: print('numero di polinomi cubici fino al grado ', 2 * M_v, ':', contanumpol)
        if ifprintbestfit: print('best alpha:', alphaP, 'deltalpha:', delta_alphaP)
        if ifprintbestfit: print('logevidence:', M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(betha0)) - \
                                 E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))

    # valuto il grado ottimale del polinomio cercando il massimo della funzione di evidence
    index = log_evidence_vP.index(max(log_evidence_vP))

    # calcolo il fit bayesiano per il valore ottimale di alpha e per il grado che massimizza la evidence.
    mN, SN, y_infer, sy_infer = bayesianpol(grid, sdata, Mv_list[index], N, alpha_vP[index], grid)

    if ifprintbestfit: print('grado ottimale', 2 * (index + 1), 'grado massimo tentato', 2 * (M_tot - 1))

    return mN, SN, y_infer, sy_infer, log_evidence_vP


def bayesianmodelprediction(grid, sdata, N, x_infer, ifprintcubic=False, ifprintmodpred=False):
    # grid e' la griglia di punti k.
    # sdata sono i valori calcolati nella simualzione con la std dev dei dati.
    # N e' il numero di dati nel fit.
    # x_infer sono i punti k dove voglio inferire il risultato.

    M_tot = 15
    # M_tot e' il numero massimo del grado del polinomio che considero
    log_evidence_vP = []
    alpha_vP = []
    betha_vP = np.zeros((M_tot))
    g_vP = np.zeros((M_tot))
    x = grid[:N, :].T * 0.13484487571168569
    y_noise = sdata[0][:N]
    Mv_list = []
    sigma_noise = sdata[1][:N]
    betha0 = (1 / sigma_noise) ** 2
    mNs = 0
    SNs = 0
    mNa = np.zeros(M_tot)
    pcont = 0
    for M_v in range(1, M_tot):  # number of parameters

        # calcolo il set di funzioni di base (armoniche cubiche) associate a questo grado M_v
        contanumpol = 1
        Phi_vP = [np.ones((x.shape[1])) * np.sqrt(betha0)]
        for i in range(1, M_v):

            s = cubicharmonics.computecubicar(i, x.T)
            contanumpol += s.shape[1]

            if ifprintcubic: print(contanumpol, i, Phi_vP[0].shape, s.shape)

            for j in range(s.shape[1]):
                Phi_vP = np.append(Phi_vP, [s[:, j] * np.sqrt(betha0)], axis=0)

                if ifprintcubic: print(contanumpol, 'in', Phi_vP.shape)

            Phi_vP = np.array(Phi_vP)

            if ifprintcubic: print(contanumpol, 'out', Phi_vP.shape)
        # contanumpol e' il numero di armoniche cubiche associate a M_v
        if M_v == 1: Phi_vP = np.array(Phi_vP)

        # calcolo gli autovalori di Phi_vP, servono per la stima di alpha ottimale
        li_vP, ei_vP = eig(np.dot(Phi_vP, Phi_vP.T))

        alpha0 = 1000
        delta_alphaP = 1
        alphaP = alpha0
        conta = 0

        # inizio il ciclo self-consistente per ottenere il valore migliore di alpha
        while abs(delta_alphaP / (alphaP + 0.1)) > 1e-10 and conta < 1.0e3:
            conta += 1
            SN_vP = np.linalg.inv(alphaP * np.identity(contanumpol) + np.dot(Phi_vP, Phi_vP.T))
            mN_vP = np.dot(np.dot(SN_vP, (Phi_vP)), y_noise * np.sqrt(betha0))
            g_vP = np.sum(li_vP.real / (alphaP + li_vP.real))
            alpha1P = g_vP / (np.dot(mN_vP.T, mN_vP))
            delta_alphaP = alpha1P - alphaP
            alphaP = alpha1P

        if (abs(delta_alphaP / (alphaP + 0.1)) > 1e-10):
            if ifprintmodpred: print('no convergence', N, x[-1], conta, delta_alphaP, alphaP, M_v)
            continue
        else:
            pcont += 1
        Mv_list.append(M_v)
        alpha_vP.append(alphaP)

        # mi preparo a calcolare la funzione di evidence per il valore ottimale di alpha
        A_vP = np.linalg.inv(SN_vP)
        E_mNs_vP = 1 / 2 * (y_noise * np.sqrt(betha0) - np.dot(Phi_vP.T, mN_vP.T)) ** 2
        E_mN_vP = E_mNs_vP.sum()
        log_evidence_vP.append(M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(betha0)) - \
                               E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))

        if ifprintmodpred: print('numero di polinomi cubici di grado ', 2 * M_v, ':', contanumpol)
        if ifprintmodpred: print('best alpha:', alphaP, 'deltalpha:', delta_alphaP)
        if ifprintmodpred: print('logevidence:', M_v / 2 * np.log(np.abs(alphaP)) + N / 2 * np.sum(np.log(betha0)) - \
                                 E_mN_vP - 1 / 2 * np.log(np.abs(np.linalg.det(A_vP))))
        mN, SN, y_infer, sy_infer = bayesianpol(grid, sdata, M_v, N, alphaP, grid)

        mNs += mN[0] * np.exp(log_evidence_vP[pcont - 1] / log_evidence_vP[0])
        if ifprintmodpred: print('determinante della matrice delle armoniche cubiche', np.linalg.det(A_vP))
        mNa[M_v] = mN[0]
        SNs += SN[0, 0] * np.exp(log_evidence_vP[pcont - 1] / log_evidence_vP[0])
        if ifprintmodpred: print('predizione a k=0 con polinomi di grado', 2 * M_v, ':', mN[0])

    sr = np.var(mNa) / M_tot ** 2 + SNs / (sum(np.exp(log_evidence_vP / log_evidence_vP[0]))) ** 2

    mr = mNs / sum(np.exp(log_evidence_vP / log_evidence_vP[0]))
    return mr, sr

