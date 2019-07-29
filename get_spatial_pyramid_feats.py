import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction


def get_spatial_pyramid_feats(image_paths, max_level, feature):
    """
    This function assumes that 'vocab_hog.npy' (for HoG) or 'vocab_sift.npy' (for SIFT)
    exists and contains an N x feature vector length matrix 'vocab' where each row
    is a kmeans centroid or visual word. This matrix is saved to disk rather than passed
    in a parameter to avoid recomputing the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path,
    :param max_level: level of pyramid,
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size'), multiplies with
        (1 / 3) * (4 ^ (max_level + 1) - 1).
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]

    # Your code here. You should also change the return value.
    imgs_n = int((1 / 3) * (4 ** (max_level + 1) - 1))
    sp = np.zeros((1500,vocab_size*imgs_n)).astype(np.float)
    n = 0

    for path in image_paths:
        img_origin = cv2.imread(path)
        height = img_origin.shape[0]
        width = img_origin.shape[1]
        idx = 0 #the number of subimage of one image
        #cut image
        for l in range(0,max_level+1):
            item_width = int(width / (2**l))
            item_height = int(height / (2**l))
            for i in range(0,2**l):
                for j in range(0,2**l):
                    subimg = img_origin[j*item_height:(j+1)*item_height,i*item_width:(i+1)*item_width,:]
                    features = feature_extraction(subimg, feature)
                    distances = []
                    for k in range(features.shape[0]):
                        distances = pdist(np.mat(features[k]),vocab)
                        indice = np.argsort(distances)[0,0]
                        sp[n,idx * vocab_size + indice] = sp[n,idx * vocab_size + indice] + 1
                    idx = idx + 1
        n = n + 1
    #sp = sp / sp.max(axis=0)

    return sp
