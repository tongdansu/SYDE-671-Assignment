#!/usr/bin/env python
# coding: utf-8

# # SYDE 671, A2

# ## 性感大野猫, 20754736

# In[37]:


# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)
 
import csv
import sys
import argparse
import numpy as np
 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
 
from skimage import io, filters, feature, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

#from helpers import cheat_interest_points, evaluate_correspondence
 
# This script
# (1) Loads and resizes images
# (2) Finds interest points in those images                 (you code this)
# (3) Describes each interest point with a local feature    (you code this)
# (4) Finds matching features                               (you code this)
# (5) Visualizes the matches
# (6) Evaluates the matches based on ground truth correspondences
 
def load_data(file_name):
    """
    1) Load stuff
    There are numerous other image sets in the supplementary data on the
    project web page. You can simply download images off the Internet, as
    well. However, the evaluation function at the bottom of this script will
    only work for three particular image pairs (unless you add ground truth
    annotations for other image pairs). It is suggested that you only work
    with the two Notre Dame images until you are satisfied with your
    implementation and ready to test on additional images. A single scale
    pipeline works fine for these two images (and will give you full credit
    for this project), but you will need local features at multiple scales to
    handle harder cases.
 
    If you want to add new images to test, create a new elif of the same format as those
    for notre_dame, mt_rushmore, etc. You do not need to set the eval_file variable unless
    you hand create a ground truth annotations. To run with your new images use
    python main.py -p <your file name>.
 
    :param file_name: string for which image pair to compute correspondence for
 
        The first three strings can be used as shortcuts to the
        data files we give you
 
        1. notre_dame
        2. mt_rushmore
        3. e_gaudi
 
    :return: a tuple of the format (image1, image2, eval file)
    """
 
    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"
 
    eval_file = "../data/NotreDame/NotreDameEval.mat"
 
    if file_name == "notre_dame":
        pass
    elif file_name == "mt_rushmore":
        image1_file = "../data/MountRushmore/Mount_Rushmore1.jpg"
        image2_file = "../data/MountRushmore/Mount_Rushmore2.jpg"
        eval_file = "../data/MountRushmore/MountRushmoreEval.mat"
    elif file_name == "e_gaudi":
        image1_file = "../data/EpiscopalGaudi/EGaudi_1.jpg"
        image2_file = "../data/EpiscopalGaudi/EGaudi_2.jpg"
        eval_file = "../data/EpiscopalGaudi/EGaudiEval.mat"
 
    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))
 
    return image1, image2, eval_file
 
def main():
    """
    Reads in the data,
 
    Command line usage: python main.py [-a | --average_accuracy] -p | --pair <image pair name>
 
    -a | --average_accuracy - flag - if specified, will compute your solution's
    average accuracy on the (1) notre dame, (2) mt. rushmore, and (3) episcopal
    guadi image pairs
 
    -p | --pair - flag - required. specifies which image pair to match
 
    """
 
    # create the command line parser
    parser = argparse.ArgumentParser()
 
    parser.add_argument("-a", "--average_accuracy", help="Include this flag to compute the average accuracy of your matching.")
    parser.add_argument("-p", "--pair", required=True, help="Either notre_dame, mt_rushmore, or e_gaudi. Specifies which image pair to match")
 
    args = parser.parse_args()
 
    # (1) Load in the data
    image1, image2, eval_file = load_data(args.pair)
 
    # You don't have to work with grayscale images. Matching with color
    # information might be helpful. If you choose to work with RGB images, just
    # comment these two lines
    image1 = rgb2gray(image1)
    image2 = rgb2gray(image2)
     
    # make images smaller to speed up the algorithm. This parameter
    # gets passed into the evaluation code, so don't resize the images
    # except for changing this parameter - We will evaluate your code using
    # scale_factor = 0.5, so be aware of this
    scale_factor = 0.5
 
    # Bilinear rescaling
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))
 
    # width and height of each local feature, in pixels
    feature_width = 16
 
    # (2) Find distinctive points in each image. See Szeliski 4.1.1
    # !!! You will need to implement get_interest_points. !!!
 
    print("Getting interest points...")
 
    # For development and debugging get_features and match_features, you will likely
    # want to use the ta ground truth points, you can comment out the precedeing two
    # lines and uncomment the following line to do this.
 
    #(x1, y1, x2, y2) = cheat_interest_points(eval_file, scale_factor)
 
    (x1, y1) = get_interest_points(image1, feature_width)
    (x2, y2) = get_interest_points(image2, feature_width)
 
    # if you want to view your corners uncomment these next lines!
 
    # plt.imshow(image1, cmap="gray")
    # plt.scatter(x1, y1, alpha=0.9, s=3)
    # plt.show()
 
    # plt.imshow(image2, cmap="gray")
    # plt.scatter(x2, y2, alpha=0.9, s=3)
    # plt.show()
 
    print("Done!")
 
    # 3) Create feature vectors at each interest point. Szeliski 4.1.2
    # !!! You will need to implement get_features. !!!
 
    print("Getting features...")
    image1_features = get_features(image1, x1, y1, feature_width)
    image2_features = get_features(image2, x2, y2, feature_width)
 
    print("Done!")
 
    # 4) Match features. Szeliski 4.1.3
    # !!! You will need to implement match_features !!!
 
    print("Matching features...")
    matches, confidences = match_features(image1_features, image2_features)
     
    if len(matches.shape) == 1:
        print( "No matches!")
        return
     
    print("Done!")
 
 
    # 5) Visualization
 
    # You might want to do some preprocessing of your interest points and matches
    # before visualizing (e.g. only visualizing 100 interest points). Once you
    # start detecting hundreds of interest points, the visualization can become
    # crowded. You may also want to threshold based on confidence
 
    # visualize.show_correspondences produces a figure that shows your matches
    # overlayed on the image pairs. evaluate_correspondence computes some statistics
    # about the quality of your matches, then shows the same figure. If you want to
    # just see the figure, you can uncomment the function call to visualize.show_correspondences
 
     
    num_pts_to_visualize = matches.shape[0]
    print("Matches: " + str(num_pts_to_visualize))
    # visualize.show_correspondences(image1, image2, x1, y1, x2, y2, matches, filename=args.pair + "_matches.jpg")
 
    ## 6) Evaluation
    # This evaluation function will only work for the Notre Dame, Episcopal
    # Gaudi, and Mount Rushmore image pairs. Comment out this function if you
    # are not testing on those image pairs. Only those pairs have ground truth
    # available.
    #
    # It also only evaluates your top 100 matches by the confidences
    # that you provide.
    #
    # Within evaluate_correspondences(), we sort your matches in descending order
    #
    num_pts_to_evaluate = matches.shape[0]
 
    evaluate_correspondence(image1, image2, eval_file, scale_factor,
        x1, y1, x2, y2, matches, confidences, num_pts_to_visualize)
 
    return
 
if __name__ == '__main__':
    main()


# In[34]:


from skimage.feature import plot_matches
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

def show_correspondences(imgA, imgB, X1, Y1, X2, Y2, matches, mode='arrows', filename=None):
	'''
		Visualizes corresponding points between two images, either as
		arrows or dots
		mode='dots': Corresponding points will have the same random color
		mode='arrows': Corresponding points will be joined by a line
		Writes out a png of the visualization if 'filename' is not None.
	'''

	# generates unique figures so students can
	# look at all three at once
	fig, ax = plt.subplots(nrows=1, ncols=1)

	if mode == 'dots':
		print("dot visualization not implemented yet :(")

	else:
		kp1 = zip_x_y(Y1, X1)
		kp2 = zip_x_y(Y2, X2)
		matches = matches.astype(int)
		plot_matches(ax, imgA, imgB, kp1, kp2, matches, only_matches=True)

	if filename:
		plt.savefig(filename, dpi = 300)
	else:
		plt.show()

	return

def zip_x_y(x, y):
	zipped_points = []
	for i in range(len(x)):
		zipped_points.append(np.array([x[i], y[i]]))
	return np.array(zipped_points)


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    image_blurred = filters.gaussian(image, sigma)
    Iy, Ix = np.gradient(image_blurred)
    Ixx = filters.gaussian(Ix * Ix, sigma)
    Iyy = filters.gaussian(Iy * Iy, sigma)
    Ixy = filters.gaussian(Ix * Iy, sigma)
    R = Ixx * Iyy - Ixy**2 - k * (Ixx + Iyy)**2
    R_norm = (R-np.min(R))/(np.max(R)-np.min(R))
    corners = R_norm
    threshold = np.mean(R_norm)
    mask = [R_norm < threshold]
    corners[mask] = 0
    keypoints = peak_local_max(corners, threshold_rel=0.2, exclude_border=True,
                               num_peaks=2000, min_distance=feature_width//2)
    ys = keypoints[:, 0]
    xs = keypoints[:, 1]
    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

   
    for xi, yi in zip(x, y):
        crop_magnitudes = magnitudes[yi-offset +
                                     1:yi+offset+1, xi-offset+1:xi+offset+1]
        crop_angles = angles[yi-offset+1:yi+offset+1, xi-offset+1:xi+offset+1]
        if crop_magnitudes.shape != (feature_width, feature_width):
            # Crop does not satisfy size constraint, skip keypoint.
            continue

        # Create SIFT descriptor
        patches_magnitudes = np.array(np.hsplit(np.array(
            np.hsplit(crop_magnitudes, 4)).reshape(4, -1), 4)).reshape(-1, feature_width)
        patches_angles = np.array(np.hsplit(np.array(
            np.hsplit(crop_angles, 4)).reshape(4, -1), 4)).reshape(-1, feature_width)
        feature_vector = list()
        for patch_i in range(patches_magnitudes.shape[0]):
            bins = np.digitize(
                patches_angles[patch_i], np.arange(0, 360, 360 // n_bins))
            bin_vector = np.zeros(n_bins)
            for bin_i in range(0, n_bins):
                mask = np.array(bins == bin_i).flatten()
                bin_vector[bin_i] = np.sum(
                    patches_magnitudes[patch_i].flatten()[mask])
            feature_vector.append(bin_vector)
        feature_vector_norm = (np.array(feature_vector).flatten() /
                               np.array(feature_vector).flatten().sum())
        descriptors.append(feature_vector_norm)
    return np.array(descriptors)



def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - zip (python built in function)

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''
    distances = cdist(im1_features, im2_features, metric="euclidean")
    n_rows = distances.shape[0]
    idxs = np.argsort(distances, axis=1)[:, :2]
    d1 = [np.arange(n_rows), idxs[:, 0]]
    d2 = [np.arange(n_rows), idxs[:, 1]]
    NDDR = distances[d1] / distances[d2]
    matches = np.stack((d1[0], d1[1]), axis=1)
    confidences = 1 - NDDR

    return matches, confidences


# In[ ]:





# In[ ]:




