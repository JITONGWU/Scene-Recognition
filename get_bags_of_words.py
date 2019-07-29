import cv2
import numpy as np
from numpy import linalg

from distance import pdist
from feature_extraction import feature_extraction


def get_bags_of_words(image_paths, feature):
    """
    This function assumes that 'vocab.mat' exists and contains an N x feature vector
    length matrix 'vocab' where each row is a kmeans centroid or visual word. This
    matrix is saved to disk rather than passed in a parameter to avoid recomputing
    the vocabulary every run.

    :param image_paths: a N array of string where each string is an image path
    :param feature: name of image feature representation.

    :return: an N x d matrix, where d is the dimensionality of the
        feature representation. In this case, d will equal the number
        of clusters or equivalently the number of entries in each
        image's histogram ('vocab_size') below.
    """
    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    vocab_size = vocab.shape[0]
    
    # Your code here. You should also change the return value.
    bag_of_words = np.zeros((1500,vocab_size))
    n = 0 # number of image
    for path in image_paths:
        img = cv2.imread(path)[:, :, ::-1]
        features = feature_extraction(img, feature)
        if features is not None :
            distances = []
            for i in range(features.shape[0]):
                distances = pdist(np.mat(features[i]),vocab) # size of distance = (1,vocabsize)
                indice = np.argsort(distances)[0,0]
                bag_of_words[n,indice] = bag_of_words[n,indice] + 1 
        print(n)
        n = n + 1
    #bag_of_words = bag_of_words / bag_of_words.max(axis=0)
    return bag_of_words
