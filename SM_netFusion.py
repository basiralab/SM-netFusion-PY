# Main function of SM_netFusion framework for a fast and accurate classification.
# Details can be found in the original paper:
# Islem Mhiri and Islem Rekik. "Supervised Multi-topology Network
# Cross-diffusion for Population-driven Brain Network Atlas Estimation"
#


#   ---------------------------------------------------------------------

#     This file contains the implementation of three key steps of our SM_netFusion framework:
#     (1) Class-specific feature extraction and clustering,
#     (2) Class-specific supervised multi-topology network cross-diffusion  and
#     (3) Discriminative connectional biomarker identification:
#
#                 [AC1,AC2,ind] = SM_netFusion(train_data,train_Labels,Nf,displays)
#
#                 Inputs:
#
#                          train_data: ((n/5) × 4) × m × m) tensor stacking the symmetric matrices of the training subjects
#                                      n the total number of subjects
#                                      m the number of nodes
#
#                          train_Labels: ((n/5) × 4) × 1) vector of training labels (e.g., -1, 1)
#
#                          Nf: Number of selected features
#
#                          displays: Boolean variables [0, 1].
#                                    if displays = 1 ==> display(Atlas of group 1, Atlas of group 2, top features matrix and the circular graph)
#                                    if displays = 0 ==> no display
#                 Outputs:
#                         AC1: (m × m) matrix stacking the atlas of group 1
#
#                         AC2: (m × m) matrix stacking the atlas of group 2
#
#                         ind: (Nf × 1) vector stacking the indices of the top disciminative features
#
#
#     To evaluate our framework we used Leave-One-Out cross validation strategy.



#To test SM-netFusion on random data, we defined the function 'simulateData' where the size of the dataset is chosen by the user.
# ---------------------------------------------------------------------
#     Copyright 2020 Birkan Ak, Istanbul Technical University.
#     Please cite the above paper if you use this code.
#     All rights reserved.
#     """

#------------------------------------------------------------------------------


import numpy as np
import SNF_all
# import snf
import SIMLR_PY.SIMLR as SIMLR
import matplotlib.pyplot as plt


# NOTE THAT THIS IS HEAVILY REFERENCED FROM NAGFS-PY

def SM_netFusion(train_data, train_Labels, Nf, displayResults):
    XC1 = np.empty((0, train_data.shape[2], train_data.shape[2]), int)
    XC2 = np.empty((0, train_data.shape[2], train_data.shape[2]), int)
    for i in range(len(train_Labels)):
        if train_Labels[i] == 1:
            XC1 = np.append(XC1, [train_data[i, :, :]], axis=0)
        else:
            XC2 = np.append(XC2, [train_data[i, :, :]], axis=0)

    # SIMLR PART, extraction of each cluster for both classes below.
    k = np.empty((0, XC1.shape[1] * XC1.shape[1]), int)
    for i in range(XC1.shape[0]):
        k1 = np.concatenate(XC1[i])
        k = np.append(k, [k1.reshape(XC1.shape[1] * XC1.shape[1])], axis=0)

    simlr = SIMLR.SIMLR_LARGE(2, 4, 0)
    S1, F1, val1, ind1 = simlr.fit(k)
    y_pred_X1 = simlr.fast_minibatch_kmeans(F1, 2)

    kk = np.empty((0, XC2.shape[1] * XC2.shape[1]), int)
    for i in range(XC2.shape[0]):
        kk1 = np.concatenate(XC2[i])
        kk = np.append(kk, [kk1.reshape(XC2.shape[1] * XC2.shape[1])], axis=0)


    simlr = SIMLR.SIMLR_LARGE(2, 4, 0)
    S2, F2, val2, ind2 = simlr.fit(kk)
    y_pred_X2 = simlr.fast_minibatch_kmeans(F2, 2)

    # Below is the same steps as NAGFS, except we will not need the third one, so we don't create a variable and skip the last else if.

    Ca1 = np.empty((0, XC1.shape[2], XC1.shape[2]), int)
    Ca2 = np.empty((0, XC1.shape[2], XC1.shape[2]), int)

    for i in range(len(y_pred_X1)):
        if y_pred_X1[i] == 0:
            Ca1 = np.append(Ca1, [XC1[i, :, :]], axis=0)
            Ca1 = np.abs(Ca1)
            # TODO Maybe add Ln1() and Ln2() but I don't know it yet.
        elif y_pred_X1[i] == 1:
            Ca2 = np.append(Ca2, [XC1[i, :, :]], axis=0)
            Ca2 = np.abs(Ca2)

    Cn1 = np.empty((0, XC2.shape[2], XC2.shape[2]), int)
    Cn2 = np.empty((0, XC2.shape[2], XC2.shape[2]), int)

    for i in range(len(y_pred_X2)):
        if y_pred_X2[i] == 0:
            Cn1 = np.append(Cn1, [XC2[i, :, :]], axis=0)
            Cn1 = np.abs(Cn1)
        elif y_pred_X2[i] == 1:
            Cn2 = np.append(Cn2, [XC2[i, :, :]], axis=0)
            Cn2 = np.abs(Cn2)

    # SNF part

    class1 = []
    if Ca1.shape[0] > 1:
        for i in range(Ca1.shape[0]):
            class1.append(Ca1[i, :, :])
        affinity_networks = SNF_all.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC11 = SNF_all.SNF_all(affinity_networks, K=20)
        class1 = []
    else:
        AC11 = Ca1[0]
    class1 = []

    if Ca2.shape[0] > 1:
        for i in range(Ca2.shape[0]):
            class1.append(Ca2[i, :, :])
        affinity_networks = SNF_all.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC12 = SNF_all.SNF_all(affinity_networks, K=20)
        class1 = []
    else:
        AC12 = Ca2[0]

    if Cn1.shape[0] > 1:
        class1 = []
        for i in range(Cn1.shape[0]):
            class1.append(Cn1[i, :, :])
        affinity_networks = SNF_all.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC21 = SNF_all.SNF_all(affinity_networks, K=20)  # First local network atlas for C2 group
        class1 = []
    else:
        AC21 = Cn1[0]

    class1 = []
    if Cn2.shape[0] > 1:
        for i in range(Cn2.shape[0]):
            class1.append(Cn2[i, :, :])
        affinity_networks = SNF_all.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC22 = SNF_all.SNF_all(affinity_networks, K=20)  # Second local network atlas for C2 group
        class1 = []
    else:
        AC22 = Cn2[0]

    AC1 = SNF_all.SNF_all([AC11, AC12], K=20)
    AC2 = SNF_all.SNF_all([AC21, AC22], K=20)

    # 5 most discriminative connectivities are determined below and being indexed in array

    D0 = np.abs(AC1 - AC2)
    D = np.triu(D0)
    D1 = D[np.triu_indices(AC1.shape[0], 1)]
    D1 = D1.transpose()
    D2 = np.sort(D1)
    D2 = D2[::-1]
    Dif = D2[0:Nf]
    D3 = []
    for i in D1:
        D3.append(i)
    ind = []
    for i in range(len(Dif)):
        ind.append(D3.index(Dif[i]))

    # DISPLAY RESULTS PART BELOW

    coord = []
    for i in range(len(Dif)):
        for j in range(D0.shape[0]):
            for k in range(D0.shape[1]):
                if Dif[i] == D0[j][k]:
                    coord.append([j, k])

    topFeatures = np.zeros((D0.shape[0], D0.shape[1]))
    s = 0
    ss = 0
    for i in range(len(Dif) * 2):
        topFeatures[coord[i][0]][coord[i][1]] = Dif[s]
        ss += 1
        if ss == 2:
            s += 1
            ss = 0


    if displayResults == 1:
        plt.imshow(topFeatures)
        plt.title('Top features')
        plt.colorbar()
        plt.show()
        plt.imshow(AC1)
        plt.title('Atlas 1')
        plt.colorbar()
        plt.show()
        plt.imshow(AC2)
        plt.title('Atlas 2')
        plt.colorbar()
        plt.show()

    return AC1, AC2, ind
