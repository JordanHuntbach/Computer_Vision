################################################################################

# Functionality: Functions for selective search

# Origin acknowledgements: https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/

################################################################################
import os
from utils import *
import cv2

# Speed-up using multi-threads
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def create_segments(img):
    # Resize image
    ratio = 1
    if ratio < 1:
        new_height = int(img.shape[0] * ratio)
        new_width = int(img.shape[1] * ratio)
        image = cv2.resize(img, (new_width, new_height))
    else:
        image = img

    # Set input image on which we will run segmentation
    ss.setBaseImage(image)

    # Set recall vs. speed
    ss.switchToSelectiveSearchQuality()
    # ss.switchToSelectiveSearchFast()

    # Run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # Number of region proposals to return
    num_show_rects = 1000

    return_rects = []

    for i, rect in enumerate(rects):
        # Draw rectangle for region proposal till num_show_rects
        if i < num_show_rects:
            x, y, w, h = rect / ratio
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            return_rects.append((x, y, x + w, y + h))
        else:
            break

    return return_rects


if __name__ == '__main__':
    directory = "Datasets/TTBB-durham-02-10-17-sub10/left-images/"
    left_file_list = sorted(os.listdir(directory))
    for filename_left in left_file_list:
        image = cv2.imread(os.path.join(directory, filename_left), cv2.IMREAD_COLOR)
        detections = create_segments(image)
        display(draw(image, detections))
