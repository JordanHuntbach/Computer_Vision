################################################################################

# Functionality: Perform detection based on HOG feature descriptor / SVM classification
# using a very basic multi-scale, sliding window (exhaustive search) approach.

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Minor portions: based on fork from https://github.com/nextgensparx/PyBOW

################################################################################

import cv2
import os
import numpy as np
import math
import params
from utils import *
from sliding_window import *
from selective_search import *
from stereo_disparity import *

################################################################################

directory_to_cycle = "Datasets/TTBB-durham-02-10-17-sub10/left-images"

show_scan_window_process = False

################################################################################


# Load SVM from file.
def load_svm():
    try:
        svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)

        # Print some checks.
        print("svm size : ", len(svm.getSupportVectors()))
        print("svm var count : ", svm.getVarCount())

        return svm
    except:
        print("Missing files - SVM!")
        print("-- have you performed training to produce these files ?")
        exit()

################################################################################


def detect_sliding(img, svm, disparity_map):
    img = img[110:390, 135:1024]
    disparities = disparity_map[110:390, 135:1024]

    # For a range of different image scales in an image pyramid.
    current_scale = -1
    detections = []
    rescaling_factor = 1.25

    # For each re-scale of the image.
    for resized in pyramid(img, scale=rescaling_factor):

        # At the start our scale = 1, because we catch the flag value -1.
        if current_scale == -1:
            current_scale = 1
        # After this rescale downwards each time (division by re-scale factor).
        else:
            current_scale /= rescaling_factor

        rect_img = resized.copy()

        # If we want to see progress show each scale.
        if show_scan_window_process:
            cv2.imshow('current scale', rect_img)
            cv2.waitKey(10)

        # Loop over the sliding window for each layer of the pyramid (re-sized image).
        window_size = params.DATA_WINDOW_SIZE
        step = math.floor(resized.shape[0] / 16)

        if step > 0:

            # For each scan window:
            for (x, y, window) in sliding_window(resized, window_size, step_size=step):

                # If we want to see progress show each scan window.
                if show_scan_window_process:
                    cv2.imshow('current window', window)
                    cv2.waitKey(10)  # wait 10ms

                # For each window region get the HoG feature point descriptors.
                img_data = ImageData(window)
                img_data.compute_hog_descriptor()

                # Generate and classify each window by constructing a HoG histogram and passing
                # it through the SVM classifier.
                if img_data.hog_descriptor is not None:

                    retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))

                    # If we get a detection, then record it.
                    if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                        # The HOG detector returns slightly larger rectangles than the real objects,
                        # so we slightly shrink the rectangles to get a nicer output.
                        pad_w, pad_h = int(0.10 * window_size[0]), int(0.05 * window_size[1])

                        # Store rect as (x1, y1), (x2,y2) pair.
                        rect = np.float32([x + pad_w, y + pad_h, x + window_size[0] - pad_w, y + window_size[1] - pad_h])

                        # If we want to see progress show each detection, at each scale.
                        if show_scan_window_process:
                            cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                            cv2.imshow('current scale', rect_img)
                            cv2.waitKey(40)

                        rect *= (1.0 / current_scale)

                        x1 = int(rect[0])
                        y1 = int(rect[1])
                        x2 = int(rect[2])
                        y2 = int(rect[3])

                        disparity = get_object_disparity(disparities[y1:y2, x1:x2])

                        if disparity > 0:
                            distance = get_object_depth(disparity)
                            if good_human_size(distance, y2 - y1, x2 - x1):
                                detections.append((x1 + 135, y1 + 110, x2 + 135, y2 + 110))

    # For the overall set of detections (over all scales) perform non maximal suppression.
    # (i.e. remove overlapping boxes etc.)
    detections = non_max_suppression_fast(np.int32(detections), 0.4)

    return detections

################################################################################


def good_human_size(distance, height, width):
    # Real Object height (m) = Distance to Object (m) x Object height on sensor (px) / focal length (px)
    human_height = distance * height / params.focal_length

    return 1.40 < human_height < 2 and width < height

################################################################################


def detect_selective(image, svm):
    img = image[20:390, 135:1024]

    detections = []

    # For each selected region:
    for (x1, y1, x2, y2) in create_segments(img):
        window = img[y1: y2, x1: x2]

        # For each window region get the HoG feature point descriptors.
        img_data = ImageData(window)
        img_data.compute_hog_descriptor()

        # If we want to see progress show each scan window.
        if show_scan_window_process:
            cv2.imshow('current window', window)
            cv2.waitKey(10)  # wait 10ms

        # Generate and classify each window by constructing a HoG histogram and passing
        # it through the SVM classifier.
        if img_data.hog_descriptor is not None:

            retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))

            # If we get a detection, then record it.
            if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                # If we want to see progress show each scan window.
                if show_scan_window_process:
                    cv2.imshow('Detection', window)
                    cv2.waitKey(40)  # wait 10ms

                # The HOG detector returns slightly larger rectangles than the real objects,
                # so we slightly shrink the rectangles to get a nicer output.
                pad_w, pad_h = int(0.15 * (x2 - x1)), int(0.05 * (y2 - y1))

                # Store rect as (x1, y1), (x2,y2) pair.
                rect = np.float32([x1 + pad_w, y1 + pad_h, x2 - pad_w, y2 - pad_h])

                rect[0] += 135
                rect[1] += 20
                rect[2] += 135
                rect[3] += 20

                detections.append(rect)

    # For the overall set of detections (over all scales) perform non maximal suppression.
    # (i.e. remove overlapping boxes etc.)
    detections = non_max_suppression_fast(np.int32(detections), 0.4)

    return detections

################################################################################


if __name__ == '__main__':
    svm = load_svm()

    # Process all images in directory (sorted by filename).
    for filename in sorted(os.listdir(directory_to_cycle)):
        if '.png' in filename:
            image = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)

            display(draw(image, detect_selective(image, svm)))

    # Close all windows
    cv2.destroyAllWindows()

################################################################################
