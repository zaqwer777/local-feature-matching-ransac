import numpy as np
import cv2
from proj3_code.student_harris import get_gradients

def get_magnitudes_and_orientations(dx, dy):
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location. 
    Args:
    -   dx: A numpy array of shape (m,n), representing x gradients in the image
    -   dy: A numpy array of shape (m,n), representing y gradients in the image

    Returns:
    -   magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location
    -   orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from 
            -PI to PI.
 
    """
    magnitudes = []#placeholder
    orientations = []#placeholder

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    magnitudes = np.sqrt(dx ** 2 + dy ** 2)
    orientations = np.arctan2(dy, dx)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return magnitudes, orientations

def get_feat_vec(x, y, magnitudes, orientations, feature_width):
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described. The grid will extend
        feature_width/2 to the left of the "center", and feature_width/2 - 1 to the right
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram 
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
        to the feature vector left to right then row by row (reading order).  
    (3) Each feature should be normalized to unit length.
    (4) Each feature should be raised to a power less than one(use .9)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though, so feel free to try it.
    The autograder will only check for each gradient contributing to a single bin.
    

    Args:
    -   x: a float, the x-coordinate of the interest point
    -   y: A float, the y-coordinate of the interest point
    -   magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
    -   orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fv: A numpy array of shape (feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.

    A useful function to look at would be np.histogram.
    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    """
    x - feat//2    x - feat//4   x     x + feat//4    x + feat//2
    Histo_1        Histo_2             Histo_3        Histo_4
    
    Histo_5        Histo_6             Histo_7        Histo_8
                                 .
    Histo_9        Histo_10            Histo_11       Histo_12
    
    Histo_13       Histo_14            Histo_15       Histo_16
    """
    histo_mag_1 = []
    histo_ori_1 = []
    print(y, feature_width)
    for row in range(y - feature_width // 2, y - feature_width // 4):
        for col in range(x - feature_width // 2, x - feature_width // 4):
            histo_ori_1.append(orientations[row][col])
            histo_mag_1.append(magnitudes[row][col])
    histo_1 = np.histogram(histo_ori_1, bins=8, range=(-np.pi, np.pi), weights=histo_mag_1)[0]
    #print(np.reshape(histo_mag_1, (4, 4)))

    histo_mag_2 = []
    histo_ori_2 = []
    for row in range(y - feature_width // 2, y - feature_width // 4):
        for col in range(x - feature_width // 4, x):
            histo_ori_2.append(orientations[row][col])
            histo_mag_2.append(magnitudes[row][col])
    histo_2 = np.histogram(histo_ori_2, bins=8, range=(-np.pi, np.pi), weights=histo_mag_2)[0]
    #print(np.reshape(histo_mag_2, (4, 4)))

    histo_mag_3 = []
    histo_ori_3 = []
    for row in range(y - feature_width // 2, y - feature_width // 4):
        for col in range(x, x + feature_width // 4):
            histo_ori_3.append(orientations[row][col])
            histo_mag_3.append(magnitudes[row][col])
    histo_3 = np.histogram(histo_ori_3, bins=8, range=(-np.pi, np.pi), weights=histo_mag_3)[0]
    #print(np.reshape(histo_mag_3, (4, 4)))

    histo_mag_4 = []
    histo_ori_4 = []
    for row in range(y - feature_width // 2, y - feature_width // 4):
        for col in range(x + feature_width // 4, x + feature_width // 2):
            histo_ori_4.append(orientations[row][col])
            histo_mag_4.append(magnitudes[row][col])
    histo_4 = np.histogram(histo_ori_4, bins=8, range=(-np.pi, np.pi), weights=histo_mag_4)[0]
    #print(np.reshape(histo_mag_4, (4, 4)))

    histo_mag_5 = []
    histo_ori_5 = []
    for row in range(y - feature_width // 4, y):
        for col in range(x - feature_width // 2, x - feature_width // 4):
            histo_ori_5.append(orientations[row][col])
            histo_mag_5.append(magnitudes[row][col])
    histo_5 = np.histogram(histo_ori_5, bins=8, range=(-np.pi, np.pi), weights=histo_mag_5)[0]
    #print(np.reshape(histo_mag_5, (4, 4)))

    histo_mag_6 = []
    histo_ori_6 = []
    for row in range(y - feature_width // 4, y):
        for col in range(x - feature_width // 4, x):
            histo_ori_6.append(orientations[row][col])
            histo_mag_6.append(magnitudes[row][col])
    histo_6 = np.histogram(histo_ori_6, bins=8, range=(-np.pi, np.pi), weights=histo_mag_6)[0]
    #print(np.reshape(histo_mag_6, (4, 4)))

    histo_mag_7 = []
    histo_ori_7 = []
    for row in range(y - feature_width // 4, y):
        for col in range(x, x + feature_width // 4):
            histo_ori_7.append(orientations[row][col])
            histo_mag_7.append(magnitudes[row][col])
    histo_7 = np.histogram(histo_ori_7, bins=8, range=(-np.pi, np.pi), weights=histo_mag_7)[0]
    #print(np.reshape(histo_mag_7, (4, 4)))

    histo_mag_8 = []
    histo_ori_8 = []
    for row in range(y - feature_width // 4, y):
        for col in range(x + feature_width // 4, x + feature_width // 2):
            histo_ori_8.append(orientations[row][col])
            histo_mag_8.append(magnitudes[row][col])
    histo_8 = np.histogram(histo_ori_8, bins=8, range=(-np.pi, np.pi), weights=histo_mag_8)[0]
    #print(np.reshape(histo_mag_8, (4, 4)))

    histo_mag_9 = []
    histo_ori_9 = []
    for row in range(y, y + feature_width // 4):
        for col in range(x - feature_width // 2, x - feature_width // 4):
            histo_ori_9.append(orientations[row][col])
            histo_mag_9.append(magnitudes[row][col])
    histo_9 = np.histogram(histo_ori_9, bins=8, range=(-np.pi, np.pi), weights=histo_mag_9)[0]
    #print(np.reshape(histo_mag_9, (4, 4)))

    histo_mag_10 = []
    histo_ori_10 = []
    for row in range(y, y + feature_width // 4):
        for col in range(x - feature_width // 4, x):
            histo_ori_10.append(orientations[row][col])
            histo_mag_10.append(magnitudes[row][col])
    histo_10 = np.histogram(histo_ori_10, bins=8, range=(-np.pi, np.pi), weights=histo_mag_10)[0]
    #print(np.reshape(histo_mag_10, (4, 4)))

    histo_mag_11 = []
    histo_ori_11 = []
    for row in range(y, y + feature_width // 4):
        for col in range(x, x + feature_width // 4):
            histo_ori_11.append(orientations[row][col])
            histo_mag_11.append(magnitudes[row][col])
    histo_11 = np.histogram(histo_ori_11, bins=8, range=(-np.pi, np.pi), weights=histo_mag_11)[0]
    #print(np.reshape(histo_mag_11, (4, 4)))

    histo_mag_12 = []
    histo_ori_12 = []
    for row in range(y, y + feature_width // 4):
        for col in range(x + feature_width // 4, x + feature_width // 2):
            histo_ori_12.append(orientations[row][col])
            histo_mag_12.append(magnitudes[row][col])
    histo_12 = np.histogram(histo_ori_12, bins=8, range=(-np.pi, np.pi), weights=histo_mag_12)[0]
    #print(np.reshape(histo_mag_12, (4, 4)))

    histo_mag_13 = []
    histo_ori_13 = []
    for row in range(y + feature_width // 4, y + feature_width // 2):
        for col in range(x - feature_width // 2, x - feature_width // 4):
            histo_ori_13.append(orientations[row][col])
            histo_mag_13.append(magnitudes[row][col])
    histo_13 = np.histogram(histo_ori_13, bins=8, range=(-np.pi, np.pi), weights=histo_mag_13)[0]
    #print(np.reshape(histo_mag_13, (4, 4)))

    histo_mag_14 = []
    histo_ori_14 = []
    for row in range(y + feature_width // 4, y + feature_width // 2):
        for col in range(x - feature_width // 4, x):
            histo_ori_14.append(orientations[row][col])
            histo_mag_14.append(magnitudes[row][col])
    histo_14 = np.histogram(histo_ori_14, bins=8, range=(-np.pi, np.pi), weights=histo_mag_14)[0]
    #print(np.reshape(histo_mag_14, (4, 4)))

    histo_mag_15 = []
    histo_ori_15 = []
    for row in range(y + feature_width // 4, y + feature_width // 2):
        for col in range(x, x + feature_width // 4):
            histo_ori_15.append(orientations[row][col])
            histo_mag_15.append(magnitudes[row][col])
    histo_15 = np.histogram(histo_ori_15, bins=8, range=(-np.pi, np.pi), weights=histo_mag_15)[0]
    #print(np.reshape(histo_mag_15, (4, 4)))

    histo_mag_16 = []
    histo_ori_16 = []
    for row in range(y + feature_width // 4, y + feature_width // 2):
        for col in range(x + feature_width // 4, x + feature_width // 2):
            histo_ori_16.append(orientations[row][col])
            histo_mag_16.append(magnitudes[row][col])
    histo_16 = np.histogram(histo_ori_16, bins=8, range=(-np.pi, np.pi), weights=histo_mag_16)[0]
    #print(np.reshape(histo_mag_16, (4, 4)))

    fv.append(histo_1)
    fv.append(histo_2)
    fv.append(histo_3)
    fv.append(histo_4)
    fv.append(histo_5)
    fv.append(histo_6)
    fv.append(histo_7)
    fv.append(histo_8)
    fv.append(histo_9)
    fv.append(histo_10)
    fv.append(histo_11)
    fv.append(histo_12)
    fv.append(histo_13)
    fv.append(histo_14)
    fv.append(histo_15)
    fv.append(histo_16)

    fv = np.reshape(np.array(fv), (np.array(fv).shape[0] * np.array(fv).shape[1],))
    fv = np.float_power(fv / np.linalg.norm(fv), .9)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


def get_features(image, x, y, feature_width):
    """
    This function returns the SIFT features computed at each of the input points
    You should code the above helper functions first, and use them below.
    You should also use your implementation of image gradients from before. 

    Args:
    -   image: A numpy array of shape (m,n), the image
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fvs: A numpy array of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    fvs = np.zeros((x.shape[0], 128))
    ix, iy = get_gradients(image)
    mag_point, ori_point = get_magnitudes_and_orientations(ix, iy)
    for ind in range(0, x.shape[0]):
        fvs[ind] = get_feat_vec(x[ind], y[ind], mag_point, ori_point, feature_width)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fvs

