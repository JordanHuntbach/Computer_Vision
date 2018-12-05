################################################################################

# functionality: parameter settings for detection algorithm training/testing

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/nextgensparx/PyBOW

################################################################################

import cv2
import os

################################################################################
# Settings for data sets in general

master_path_to_dataset = "Datasets/"  # ** need to edit this **

# Data location - training examples
DATA_training_path_neg = os.path.join(master_path_to_dataset, "INRIAPerson/Train/neg/")
DATA_training_path_pos = os.path.join(master_path_to_dataset, "INRIAPerson/train_64x128_H96/pos/")

# Data location - testing examples
DATA_testing_path_neg = os.path.join(master_path_to_dataset, "INRIAPerson/Test/neg/")
DATA_testing_path_pos = os.path.join(master_path_to_dataset, "INRIAPerson/test_64x128_H96/pos/")

# Size of the sliding window patch / image patch to be used for classification
# (for larger windows sizes, for example from selective search - resize the
# window to this size before feature descriptor extraction / classification)
DATA_WINDOW_SIZE = [64, 128]

# The maximum left/right, up/down offset to use when generating samples for training
# that are centred around the centre of the image
DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 3

# Number of sample patches to extract from each negative training example
DATA_training_sample_count_neg = 10

# Number of sample patches to extract from each positive training example
DATA_training_sample_count_pos = 5

# Class names - N.B. ordering of 0, 1 for neg/pos = order of paths
DATA_CLASS_NAMES = {
    "other": 0,
    "pedestrian": 1
}

################################################################################
# Camera details.

focal_length = 399.9745178222656  # pixels
baseline_distance = 0.2090607502  # meters

################################################################################
# Settings for HOG approaches

HOG_SVM_PATH = "svm_rbf.xml"

HOG_SVM_kernel = cv2.ml.SVM_RBF  # see opencv manual for other options
HOG_SVM_max_training_iterations = 500  # stop training after max iterations

################################################################################
