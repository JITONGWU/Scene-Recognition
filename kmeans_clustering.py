import numpy as np
from distance import pdist
from sklearn.cluster import KMeans

def kmeans_clustering(all_features, vocab_size, epsilon, max_iter):
    """
    The function kmeans implements a k-means algorithm that finds the centers of vocab_size clusters
    and groups the all_features around the clusters. As an output, centroids contains a
    center of the each cluster.

    :param all_features: an N x d matrix, where d is the dimensionality of the feature representation.
    :param vocab_size: number of clusters.
    :param epsilon: When the maximum distance between previous and current centroid is less than epsilon,
        stop the iteration.
    :param max_iter: maximum iteration of the k-means algorithm.

    :return: an vocab_size x d array, where each entry is a center of the cluster.
    """

    # Your code here. You should also change the return value.
    
    n = all_features.shape[0]
    dim = all_features.shape[1]
    
    mean = np.mean(all_features, axis = 0)
    std = np.std(all_features, axis = 0)
    centers = np.random.randn(vocab_size,dim)*std + mean
    
    centers_old = np.zeros(centers.shape)
    centers_new = centers
    
    error = np.linalg.norm(centers_new - centers_old)
    
    distances = np.zeros((n,vocab_size))
    clusters = np.zeros(n)
    
    iteration = 0
    
    while error >= epsilon and iteration < max_iter :
        for i in range(vocab_size):
            distances[:,i] = np.linalg.norm(all_features - centers_new[i], axis = 1)
        clusters = np.argmin(distances,axis = 1)
        centers_old = centers_new
        for i in range(vocab_size):
            centers_new[i] = np.mean(all_features[clusters == i],axis = 0)
                
        error = np.linalg.norm(centers_new - centers_old)
        iteration = iteration + 1
    nans = []
    for i in range(vocab_size):
        if np.isnan(centers_new[i,0]):
            nans.append(i)
    centers_new = np.delete(centers_new, nans, axis=0)
    
    return centers_new