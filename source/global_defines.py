import numpy as np

# globals defined here
#------------------------
def init():
    global HOUGH_CIRCLES_DP
    global HOUGH_CIRCLES_MD
    global HOUGH_CIRCLES_P1
    global HOUGH_CIRCLES_P2
    global HOUGH_CIRCLES_MINR
    global HOUGH_CIRCLES_MAXR
    
    global DEBUG_MODE
    global INCLUDE_HCSLIDER
    global INCLUDE_GSLIDER
    
    global GAUSSIAN_BLUR_KSIZE
    global CANNY_LOW_THRESHOLD
    global CANNY_HIGH_THRESHOLD
    global HOUGH_LINESP_RHO
    global HOUGH_LINESP_THETA
    global HOUGH_LINESP_THRESHOLD
    
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
    
    HOUGH_CIRCLES_DP = 1
    HOUGH_CIRCLES_MD = 260
    HOUGH_CIRCLES_P1 = 30
    HOUGH_CIRCLES_P2 = 100
    HOUGH_CIRCLES_MINR = 0
    HOUGH_CIRCLES_MAXR = 0

    DEBUG_MODE = 0
    INCLUDE_HCSLIDER = 0
    INCLUDE_GSLIDER = 0

    GAUSSIAN_BLUR_KSIZE = 5
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    HOUGH_LINESP_RHO = 1                # distance resolution in pixels of the Hough grid
    HOUGH_LINESP_THETA = np.pi/180      # angular resolution in radians of the Hough grid
    HOUGH_LINESP_THRESHOLD = 50        # minimum number of votes (intersections in Hough grid cell)

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
    #------------------------