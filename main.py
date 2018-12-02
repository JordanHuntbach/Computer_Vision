import statistics

import hog_detector
import stereo_disparity
import os
import cv2

master_path_to_dataset = "Datasets/TTBB-durham-02-10-17-sub10"

directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

left_file_list = sorted(os.listdir(full_path_directory_left))

focal_length = 399.9745178222656  # pixels
baseline_distance = 0.2090607502  # meters

if __name__ == '__main__':
    svm = hog_detector.load_svm()
    for filename_left in left_file_list:

        # From the left image filename get the corresponding right image.
        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

        # Print to standard output as requested.
        print(filename_left)
        print(filename_right)

        # Load the image.
        image = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)

        # Perform pedestrian detection.
        detections = hog_detector.detect(image, svm)
        output_img = hog_detector.draw(image, detections)

        # Calculate disparities.
        disparities = stereo_disparity.calculate_disparity(filename_left)

        stereo_disparity.display_disparity(disparities)

        distances = []
        for detection in detections:
            local_disparities = disparities[detection[1]:detection[3], detection[0]:detection[2]]
            local_disparities = local_disparities.flatten()
            centre_x = int(detection[0] + (detection[2] - detection[0]) // 2)
            centre_y = int(detection[1] + (detection[3] - detection[1]) // 2)

            disparity = statistics.median(local_disparities)
            if disparity > 0:
                # Depth = focal length * baseline distance / disparity.
                depth = focal_length * baseline_distance / disparity

                cv2.putText(output_img, "%.2fm" % depth, (centre_x, centre_y),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                distances.append(depth)
            else:
                distances.append(-1)

        hog_detector.display(output_img)

        smallest_distance = None
        for value in distances:
            if value != -1:
                if smallest_distance is None or value < smallest_distance:
                    smallest_distance = value
        if smallest_distance is None:
            smallest_distance = 0
        print("Nearest detected scene object (%.2fm)" % smallest_distance)
        print()
