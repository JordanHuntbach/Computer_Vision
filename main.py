from imutils.object_detection import non_max_suppression

import hog_detector
import stereo_disparity
import os
import cv2
import params
from utils import *

############################################################################

master_path_to_dataset = "Datasets/TTBB-durham-02-10-17-sub10"

directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

left_file_list = sorted(os.listdir(full_path_directory_left))

skip_forward_file_pattern = ""

############################################################################
# OpenCV default HoG pedestrian detection.


def built_in(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    copy = image.copy()

    # Detect people in the image.
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Apply non-maximal suppression to the bounding boxes using a fairly large overlap
    # threshold to try to maintain overlapping boxes that are still people.
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # Draw the bounding boxes.
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(copy, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Show the output images.
    cv2.imshow("Built in detection", copy)


############################################################################
# Main function performs pedestrian detection using HoG detector, and then
# calculates the distance to each detected object.


if __name__ == '__main__':
    # Load the SVM.
    svm = hog_detector.load_svm()

    for filename_left in left_file_list:
        if '.png' in filename_left:
            # Skip forward to start a file we specify by timestamp (if this is set).
            if (len(skip_forward_file_pattern) > 0) and not (skip_forward_file_pattern in filename_left):
                continue
            elif (len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left):
                skip_forward_file_pattern = ""

            # From the left image filename get the corresponding right image.
            filename_right = filename_left.replace("_L", "_R")
            full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
            full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

            # Load the image.
            image = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)

            # Use CLAHE to improve contrast - didn't seem very effective.

            # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # lab_planes = cv2.split(lab)
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            # lab_planes[0] = clahe.apply(lab_planes[0])
            # lab = cv2.merge(lab_planes)
            # image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Calculate disparity map and display it in a window.
            disparities = stereo_disparity.calculate_disparity(filename_left)

            # Perform object detection.
            pedestrians, vehicles = hog_detector.detect_sliding(image, svm, disparities)
            output_img = draw(image, pedestrians)

            # Get the distance to each object, and draw it on the output image.
            distances = []
            for detection in pedestrians:
                x1 = detection[0]
                y1 = detection[1]
                x2 = detection[2]
                y2 = detection[3]

                disparity = stereo_disparity.get_object_disparity(disparities[y1:y2, x1:x2])
                if disparity > 0:
                    depth = stereo_disparity.get_object_depth(disparity)
                    cv2.putText(output_img, "P: %.2fm" % depth, (detection[0], detection[3] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    distances.append(depth)

            for detection in vehicles:
                x1 = detection[0]
                y1 = detection[1]
                x2 = detection[2]
                y2 = detection[3]

                disparity = stereo_disparity.get_object_disparity(disparities[y1:y2, x1:x2])
                if disparity > 0:
                    depth = stereo_disparity.get_object_depth(disparity)
                    cv2.putText(output_img, "V: %.2fm" % depth, (detection[0], detection[3] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    distances.append(depth)

            # Display output images.
            stereo_disparity.display_disparity(disparities)
            display(output_img)

            # Find the distance to the nearest object, to send to the standard output.
            smallest_distance = None
            for value in distances:
                if smallest_distance is None or value < smallest_distance:
                    smallest_distance = value
            if smallest_distance is None:
                smallest_distance = 0

            # Print to standard output as requested.
            print(filename_left)
            print(filename_right + " : nearest detected scene object (%.2fm)" % smallest_distance)
            print()
