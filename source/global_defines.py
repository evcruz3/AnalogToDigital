import numpy as np
from random import randint

# globals defined here
#------------------------
def init():
    global input_stream
    global MIN_ANGLE
    global MAX_ANGLE
    global MIN_VALUE
    global MAX_VALUE
    global ANGLE_RANGE
    global UNIT
    
    global tracker_colors
    global tracker_types
    global tracker_type

    global HOUGH_CIRCLES_DP
    global HOUGH_CIRCLES_MD
    global HOUGH_CIRCLES_P1
    global HOUGH_CIRCLES_P2
    global HOUGH_CIRCLES_MINR
    global HOUGH_CIRCLES_MAXR
    
    global DEBUG_MODE
    global INCLUDE_HCSLIDER
    global INCLUDE_GSLIDER
    global INCLUDE_HSVSLIDER
    
    global GAUSSIAN_BLUR_KSIZE
    global CANNY_LOW_THRESHOLD
    global CANNY_HIGH_THRESHOLD
    global HOUGH_LINESP_RHO
    global HOUGH_LINESP_THETA
    global HOUGH_LINESP_THRESHOLD
    
    global HSV_HUL
    global HSV_SAL
    global HSV_VAL
    global HSV_HUH
    global HSV_SAH
    global HSV_VAH
    
    global MEDIAN_BLUR_KSIZE
    global ERODE_KERNEL
    global DILATE_KERNEL
    
    global dp
    global md
    global p1
    global p2
    global minR
    global maxR
    
    global gb
    global mb
    global erode
    global dilate
    
    global hl
    global hh
    global sl
    global sh
    global vl
    global vh
    
    MIN_VALUE = 0
    HOUGH_CIRCLES_DP = 1
    HOUGH_CIRCLES_MD = 260
    HOUGH_CIRCLES_P1 = 30
    HOUGH_CIRCLES_P2 = 100
    HOUGH_CIRCLES_MINR = 0
    HOUGH_CIRCLES_MAXR = 0

    DEBUG_MODE = 0
    INCLUDE_HCSLIDER = 0
    INCLUDE_GSLIDER = 0
    INCLUDE_HSVSLIDER = 0

    GAUSSIAN_BLUR_KSIZE = 5
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    HOUGH_LINESP_RHO = 1                # distance resolution in pixels of the Hough grid
    HOUGH_LINESP_THETA = np.pi/180      # angular resolution in radians of the Hough grid
    HOUGH_LINESP_THRESHOLD = 50        # minimum number of votes (intersections in Hough grid cell)

    HSV_HUL = 0
    HSV_SAL = 145
    HSV_VAL = 61
    HSV_HUH = 176
    HSV_SAH = 255
    HSV_VAH = 255

    MEDIAN_BLUR_KSIZE = 5
    ERODE_KERNEL = 3
    DILATE_KERNEL = 3
    
    tracker_colors = [(randint(0, 255), randint(0, 255), randint(0, 255)), (randint(0, 255), randint(0, 255), randint(0, 255))]
    
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    
    dp = 'dp'
    md = 'minDist'
    p1 = 'param1'
    p2 = 'param2'
    minR = 'minRadius'
    maxR = 'maxRadius'

    gb = 'Gaussian Blur'
    mb = 'Median Blur'
    erode = 'Erode Kernel Value'
    dilate = 'Dilate Kernel Value'
    
    hl = 'H Low'
    hh = 'H High'
    sl = 'S Low'
    sh = 'S High'
    vl = 'V Low'
    vh = 'V High'
#------------------------
