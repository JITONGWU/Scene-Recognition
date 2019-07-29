# Scene-Recognition
Scene Recognition with Bag of Words - Machine Learning for Computer Vision


Overview
We will perform scene recognition with the bag of words method. We will classify scenes into one of 15 categories by training and testing on the 15 scene database (Lazebnik et al. 2006).

Task: Implement three scene recognition schemes:

 - Build vocabulary by k-means clustering(feature_extraction.py, kmeans_clustering.py).
 - Principle component analysis(PCA) for vocabulary(get_freatures_from_pca.py).
 - Bag of words representation of scenes(get_bags_of_words.py, get_spatial_pyramid_feats.py)
 - Multi-class SVM(svm_classify.py).
