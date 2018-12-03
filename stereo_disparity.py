#####################################################################

# Example : Load, display and compute SGBM disparity for a set of rectified
# stereo images from a  directory structure of left-images / right-images
# with files named DATE_TIME_STAMP_{L|R}.png

# Basic illustrative python script for use with provided stereo datasets.

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np

# Where is the data ? - set this to where you have it
master_path_to_dataset = "Datasets/TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

#####################################################################

# Resolve full directory location of data set for left / right images.
full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

# Get a list of the left image files and sort them (by timestamp in filename).
left_file_list = sorted(os.listdir(full_path_directory_left))

# Setup the disparity stereo processor to find a maximum of 128 disparity values.
# (Adjust parameters if needed - this will effect speed to processing)

# Uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

crop_disparity = False  # Display full or cropped disparity image.
pause_playback = False  # Pause until key press after each image.


def calculate_disparity(filename_left):

    # From the left image filename get the corresponding right image.
    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # Check the file is a PNG file (left) and check a corresponding right image actually exists.
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):

        # Read left and right images and display in windows.
        # N.B. Despite one being greyscale, both are in fact stored as 3-channel RGB images so load both as such.
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # Remember to convert to greyscale (as the disparity matching works on greyscale).
        # N.B. Need to do for both as both are 3-channel images.
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Perform pre-processing - raise to the power, as this subjectively appears to improve subsequent
        # disparity calculation.
        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')

        # Compute disparity image from undistorted and rectified stereo images that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16).
        disparity = stereoProcessor.compute(grayL, grayR)

        # Filter out noise and speckles (adjust parameters as needed).
        dispNoiseFilter = 5  # Increase for more aggressive filtering.
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

        # Scale the disparity to 8-bit for viewing:
        # Divide by 16 and convert to 8-bit image (then range of values should be 0 -> max_disparity)
        # but in fact is (-1 -> max_disparity - 1) so we fix this also using a initial threshold between
        # 0 and max_disparity as disparity=-1 means no disparity available.
        _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        return disparity_scaled

    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

#####################################################################


def display_disparity(disparity):
    global crop_disparity
    global pause_playback

    # Crop disparity to chop out left part where there are with no disparity, as this area is not seen
    # by both cameras and also chop out the bottom area (where we see the front of car bonnet).
    if crop_disparity:
        width = np.size(disparity, 1)
        disparity = disparity[0:390, 135:width]

    # Display image (scaling it to the full 0->255 range based on the number of disparities in use for
    # the stereo part).
    cv2.imshow("Disparity", (disparity * (256. / max_disparity)).astype(np.uint8))

    # Keyboard input for exit (as standard) and cropping"
    # exit - x
    # crop - c
    # pause - space
    key = cv2.waitKey(40 * (not pause_playback)) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
    if key == ord('x'):  # exit
        return
    elif key == ord('c'):  # crop
        crop_disparity = not crop_disparity
    elif key == ord(' '):  # pause (on next frame)
        pause_playback = not pause_playback

#####################################################################


if __name__ == '__main__':
    for filename_left in left_file_list:
        disparity = calculate_disparity(filename_left)
        display_disparity(disparity)

    # Close all windows
    cv2.destroyAllWindows()
