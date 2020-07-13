import cv2
import numpy as np
import matplotlib.pyplot as plt
from proj3_code.utils import load_image
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.
    
    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """

    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################

    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.dot(kernel, np.transpose(kernel))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return kernel


def my_filter2D(image, filt, bias=0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = np.zeros((image.shape[0], image.shape[1]))

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################
    #pad_image = np.zeros((image.shape[0] + 1, image.shape[1] + 1))
    pad_image = np.pad(image, 1, mode='constant')
    pad_image[1: image.shape[0] + 1, 1: image.shape[1] + 1] = image
    height = filt.shape[0] // 2
    width = filt.shape[1] // 2
    if filt.shape[0] == 1:
        height = 1
    if filt.shape[1] == 1:
        width = 1
    for i in range(height, pad_image.shape[0] - height):
        #print("I index:", i)
        for j in range(width, pad_image.shape[1] - width):
            #print("J index:", j)
            sum = 0
            for m in range(filt.shape[0]):
                for n in range(filt.shape[1]):
                    #print("For i and j:", i, j, "Index:", i - height + m, j - width + n)
                    sum += filt[m][n] * pad_image[i - height + m][j - width + n]
            conv_image[i - 1][j - 1] = sum
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image


def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """

    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    ix = my_filter2D(image, sobel_x)
    #print(ix)
    iy = my_filter2D(image, sobel_y)
    #print(iy)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return ix, iy


def remove_border_vals(image, x, y, c, window_size=16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: A numpy array of shape (N,) containing the x coordinate of each pixel
    -   y: A numpy array of shape (N,) containing the y coordinate of each pixel
    -   c: A numpy array of shape (N,) containing the confidences of each pixel
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        (set this to 16 for unit testing). treat the center point of this window as the bottom right
        of the center most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-border_vals_removed,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-border_vals_removed,) containing y-coordinates of interest points
    -   c: numpy array of shape (N-border_vals_removed,) containing the confidences of each pixel
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################

    #print(image)
    #print(x)
    #print(y)
    #print(c)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c


def second_moments(ix, iy, ksize=7, sigma=10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: A numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################
    filt = get_gaussian_kernel(ksize, sigma)
    sx = ix * ix
    sx2 = my_filter2D(sx, filt)
    sy = iy * iy
    sy2 = my_filter2D(sy, filt)
    sxsy = my_filter2D(ix * iy, filt)

    if ksize == 1:
        sx2 = sx
        sy2 = sy
        sxsy = ix * iy

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy


def corner_response(sx2, sy2, sxsy, alpha=.05):
    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]


    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################
    M = np.array(([sx2, sxsy], [sxsy, sy2]))
    det_M = sx2 * sy2 - sxsy * sxsy
    trace = alpha * (np.trace(M) * np.trace(M))
    R = det_M - trace

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R


def non_max_suppression(R, neighborhood_size=7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """

    R_local_pts = None

    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################
    median = np.median(R)
    R = np.where(R < median, 0, R)
    R_local_pts = maximum_filter(R, neighborhood_size, mode='constant')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts


def get_interest_points(image, n_pts=1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences: numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, R_local_pts, confidences = np.zeros((n_pts,)), np.zeros((n_pts,)), None, np.zeros((n_pts,))

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################
    ix, iy = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy)
    R = corner_response(sx2, sy2, sxsy)
    R_local_pts = non_max_suppression(R)
    R_ind = np.reshape(R_local_pts, (R_local_pts.shape[0] * R_local_pts.shape[1]))
    sort_arr = np.argsort(R_ind)[::-1]
    if len(sort_arr) <= n_pts:
        for ind in range(len(sort_arr)):
            x[ind] = sort_arr[ind] % image.shape[1]
            y[ind] = sort_arr[ind] // image.shape[1]
            #confidences[ind] = R_local_pts[int(x[ind])][int(y[ind])]
    else:
        for ind in range(n_pts):
            x[ind] = sort_arr[ind] % image.shape[1]
            y[ind] = sort_arr[ind] // image.shape[1]
            print(x[ind])
            #confidences[ind] = R_local_pts[int(x[ind])][int(y[ind])]
    #remove_border_vals(image, x, y, confidences)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return x, y, R_local_pts, confidences


if __name__ == '__main__':
    dummy_image = np.array(
        [
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
        ])

    #x, y, R, confidence = get_interest_points(dummy_image)
    #xy = [(x[i], y[i]) for i in range(len(x))]

    #image1 = load_image('../data/Notre Dame/921919841_a30df938f2_o.jpg')
    #image2 = load_image('../data/Notre Dame/4191453057_c86028ce1f_o.jpg')

    #scale_factor = 0.5
    #image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
    #image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
    #image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    #image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    #x1, y1, R1, _ = get_interest_points(image1_bw, 1500)
    #x2, y2, R2, _ = get_interest_points(image2_bw, 1500)
