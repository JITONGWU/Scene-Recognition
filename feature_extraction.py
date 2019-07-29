import cv2
import numpy as np
from get_image_paths import get_image_paths


def feature_extraction(img, feature):
    """
    This function computes defined feature (HoG, SIFT) descriptors of the target image.

    :param img: a height x width x channels matrix,
    :param feature: name of image feature representation.

    :return: a N x feature_size matrix.
    """

    if feature == 'HoG':
        # HoG parameters
        win_size = (32, 32)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        deriv_aperture = 1
        win_sigma = 4
        histogram_norm_type = 0
        l2_hys_threshold = 2.0000000000000001e-01
        gamma_correction = 0
        nlevels = 64
        
        # Your code here. You should also change the return value.

        hog = cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins,deriv_aperture,win_sigma,histogram_norm_type,l2_hys_threshold,gamma_correction,nlevels)

        dsize = hog.getDescriptorSize()
        descripters = hog.compute(img,winStride=(32,32),padding=(0,0))
        descripters = descripters.reshape(-1,dsize)


    elif feature == 'SIFT':
        sift = cv2.xfeatures2d.SIFT_create()
        descripters = []
        height= img.shape[0]
        width = img.shape[1]
        split1 = np.array_split(img, width/20, axis=1)
        for split in split1:
            split2 =np.array_split(split, height/20, axis=0)
            for ig in split2:
                keypoints, descripter = sift.detectAndCompute(ig,None)
                if descripter is not None:
                    descripters.append(descripter)
        if len(descripters) > 0:
            descripters = np.vstack(descripters)
        else: 
            return None
    return descripters


if __name__ == '__main__':
    
    train_image_paths, test_image_paths, train_labels, test_labels = \
    get_image_paths(data_path, categories, num_train_per_cat)
    
    for path in train_image_paths:
        img = cv2.imread(train_image_paths[0])[:, :, ::-1]  # 이미지 읽기

        features = feature_extraction(img, 'SIFT')  # 이미지에서 feature 추출
