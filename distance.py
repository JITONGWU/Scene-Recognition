import numpy as np
from get_image_paths import get_image_paths
from feature_extraction import feature_extraction
import cv2
def pdist(a, b):
    a_square = np.einsum('ij,ij->i', a, a)
    a_square = np.tile(np.reshape(a_square, [a.shape[0], 1]), [1, b.shape[0]])
    
    b_square = np.einsum('ij,ij->i', b, b)
    b_square = np.tile(np.reshape(b_square, [1, b.shape[0]]), [a.shape[0], 1])


    ab = np.dot(a, b.T)

    dist = a_square + b_square - 2 * ab

    dist = dist.clip(min=0)

    return np.sqrt(dist)

if __name__ == '__main__':
    a  = np.array(np.mat('1 2 3'))
    b = np.array(np.mat('3 1;2 1'))
    print(pdist(a,b))
    '''
    vocab = np.load('vocab_hog.npy')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)
    feature = 'HoG'
    for path in train_image_paths:
        img = cv2.imread(path)[:, :, ::-1]
        features = feature_extraction(img, feature)
        distances = []
        for i in range(features.shape[0]):
            if i == 0 :
                xxx = features[0]
                
            print(xxx.shape)
            print(a.shape)
            xxx = np.mat(xxx).transpose()
            print(xxx.shape)
            distances = pdist(features[i],vocab)
            
'''
        
        