import numpy as np


def get_features_from_pca(feat_num, feature):
    """
    This function loads 'vocab_sift.npy' or 'vocab_hog.npg' file and
    returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
    :param feature: 'Hog' or 'SIFT'

    :return: an N x feat_num matrix
    """

    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    # Your code here. You should also change the return value.
    print(vocab.shape)
    #normalization
    avr = np.mean(vocab,axis = 0)
    vocab_norm = vocab - avr
    #calculate covmatrix
    covMat = np.cov(vocab_norm,rowvar = 0) #every row is a sample
    #calculate eigenvalue and eigenvector
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    #remain feat_num eigvects
    i = np.argsort(eigVals)
    print(eigVals.shape)
    feat_num_indice = i[-1:-(feat_num+1):-1]
    feat_num_eigVect = eigVects[:,feat_num_indice]
    reduction_vocab = vocab_norm * feat_num_eigVect
    

    return reduction_vocab


