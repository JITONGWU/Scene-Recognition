import numpy as np
from sklearn import svm


def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
    """
    This function should train a linear SVM for every category (i.e., one vs all)
    and then use the learned linear classifiers to predict the category of every
    test image. Every test feature will be evaluated with all 15 SVMs and the
    most confident SVM will 'win'.

    :param train_image_feats: an N x d matrix, where d is the dimensionality of the feature representation.
    :param train_labels: an N array, where each entry is a string indicating the ground truth category
        for each training image.
    :param test_image_feats: an M x d matrix, where d is the dimensionality of the feature representation.
        You can assume M = N unless you've modified the starter code.
    :param kernel_type: SVM kernel type. 'linear' or 'RBF'

    :return:
        an M array, where each entry is a string indicating the predicted
        category for each test image.
    """

    categories = np.unique(train_labels)

    # Your code here. You should also change the return value.
    if kernel_type == 'RBF':
        
        prediction = []
        rbf_clf = svm.SVC(kernel = 'rbf')
        rbf_clf.fit(train_image_feats,train_labels)
        df = rbf_clf.decision_function(test_image_feats)
        for i in range(df.shape[0]):
            max_index = list(df[i]).index(max(df[i]))
            prediction.append(categories[max_index])
        
    elif kernel_type == 'linear':
        
        prediction = []
        lin_clf = svm.LinearSVC()
        lin_clf.fit(train_image_feats,train_labels)
        df = lin_clf.decision_function(test_image_feats)
        for i in range(df.shape[0]):
            max_index = list(df[i]).index(max(df[i]))
            prediction.append(categories[max_index])

    prediction = np.array(prediction)
    return prediction