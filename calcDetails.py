from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from math import pi

# Global variables
src_image = np.zeros((600,600), np.uint8)
src_gray = np.zeros((600,600), np.uint8)
source_window = 'details_number'

def draw_text_bg(img, text,
          font=cv.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=3,
          font_thickness=5,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

# Callback function for the tracker, `threshold` is the position value
def thresh_callback(threshold):
    global src_image
    image_width = src_image.shape[0]
    # average hole's size. S = pi * r^2, Perim = 2 * pi * r
    hole_radius = image_width / 350
    
    perimeter_avg = 2 * pi * hole_radius
    perim_min = perimeter_avg * 0.7
    perim_max = perimeter_avg * 1.3

    color_cntr = (255, 100, 255)

    # Detect edges using Canny    
    global src_gray
    threshold_output = cv.Canny(src_gray, threshold, 255)

    # Find contours
    contours, hierarchy = cv.findContours(threshold_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    
    draw_image = src_image.copy() #threshold_output.copy()
    cnt = 0
    
    for i in range(len(contours)):
        perimeter = cv.arcLength(contours[i], True)
        area_t = (perimeter/2)**2/pi
        area = cv.contourArea(contours[i])
        if perimeter > perim_min and perimeter < perim_max \
            and area > area_t*0.6 and area < area_t*1.3:
            cnt += 1
            cv.drawContours(image=draw_image, contours=contours, contourIdx=i, color=color_cntr, \
                thickness=2, lineType=cv.LINE_8, hierarchy=hierarchy, maxLevel=0)
    draw_text_bg(draw_image, f'{cnt}')
    
    # Show in a window
    global source_window
    src_scaled = cv.resize(draw_image, (800, 800))
    cv.imshow(source_window, src_scaled)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # Load source image
    parser = argparse.ArgumentParser(description='Calculate details on image.')
    parser.add_argument('input', type=str, help='Path to input image. E.g. "images/img1.jpg"')
    args = parser.parse_args()

    src_image = cv.imread(cv.samples.findFile(args.input))
    
    if (src_image is None):
        print('Could not open image', args.input)
        exit(0)

    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)

    # Create Window    
    cv.namedWindow(source_window)
    src_scaled = cv.resize(src_image, (800, 800))
    cv.imshow(source_window, src_scaled)

    max_threshold = 255
    threshold = 127 # initial threshold

    cv.createTrackbar('Threshold:', source_window, threshold, max_threshold, thresh_callback)

    thresh_callback(threshold)

    cv.waitKey()
