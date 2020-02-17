#!/usr/bin/env python
# import the necessary packages
import numpy as np
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import logging as log
import cv2
import time
from slider import *

# globals defined here
#------------------------
DEBUG_MODE = 0
INCLUDE_HCSLIDER = 0
INCLUDE_GSLIDER = 0

GAUSSIAN_BLUR_KSIZE = 5
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_LINESP_RHO = 1                # distance resolution in pixels of the Hough grid
HOUGH_LINESP_THETA = np.pi/180      # angular resolution in radians of the Hough grid
HOUGH_LINESP_THRESHOLD = 50        # minimum number of votes (intersections in Hough grid cell)

HOUGH_CIRCLES_DP = 1
HOUGH_CIRCLES_MD = 260
HOUGH_CIRCLES_P1 = 30
HOUGH_CIRCLES_P2 = 100
HOUGH_CIRCLES_MINR = 0
HOUGH_CIRCLES_MAXR = 0

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



def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-d", "--debug", action='store_true', default=False)
    args.add_argument("-chs", "--hcslider", action='store_true', default=False, help='Enables the HC slider')
    args.add_argument("-gs", "--gslider", action='store_true', default=False, help='Enables the gray slider')
                    

    return parser

def findDial(tmp_gray):
    gray = cv2.GaussianBlur(tmp_gray,(GAUSSIAN_BLUR_KSIZE, GAUSSIAN_BLUR_KSIZE),0)
    
    ret, gray = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    
    #kernel = np.ones((3,3),np.uint8)
    #gray = cv2.dilate(gray,kernel,iterations = 3)
    
    edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    
    lines = cv2.HoughLinesP (edges, 
                            HOUGH_LINESP_RHO, 
                            HOUGH_LINESP_THETA, 
                            HOUGH_LINESP_THRESHOLD, 
                            np.array([]),
                            min_line_length, 
                            max_line_gap)
    
    
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                 #perimeter.append(abs(x2-x1) + abs(y2-y1))
                 cv2.line(gray,(x1,y1),(x2,y2),(125,125,125),5)
                 #cv2.line(output,(x1,y1),(x2,y2),(255,0,0),5)
    else:
        print("No Line Detected")
    
    
    cv2.imshow("line_gray", gray)
    
    return lines
    
    '''perimeter=[]

    if lines.size > 0:
        for line in lines:
            for x1,y1,x2,y2 in line:
                 perimeter.append(abs(x2-x1) + abs(y2-y1))
                 #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            
        #detection of longest line (dial)
        ind = np.argmax(perimeter)
    
        return lines[ind]
    else:
        return 0'''


def main():
    # access global variables
    global HOUGH_CIRCLES_DP
    global HOUGH_CIRCLES_MD
    global HOUGH_CIRCLES_P1
    global HOUGH_CIRCLES_P2
    global HOUGH_CIRCLES_MINR
    global HOUGH_CIRCLES_MAXR
    global INCLUDE_HCSLIDER
    global INCLUDE_GSLIDER

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    if(args.debug):
        DEBUG_MODE = 1
    if(args.hcslider):
        INCLUDE_HCSLIDER = 1
    if(args.gslider):
        INCLUDE_GSLIDER = 1

    if args.input == 'cam':
        input_stream = '/dev/video0'
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream) # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices

    #Set Width and Height 
    # cap.set(3,1280)
    # cap.set(4,720)

    # The above step is to set the Resolution of the Video. The default is 640x480.
    # This example works with a Resolution of 640x480.

    #must find a way to localize the location for HoughCircles
    y = 0;
    x = 0;
    r = 0;
    #n = 0;
    
    if INCLUDE_HCSLIDER:
        HC_slider = make_slider('Hough Circles Slider')
        HC_slider.makeNewSlideObject(dp, 1, 5, HOUGH_CIRCLES_DP)
        HC_slider.makeNewSlideObject(md, 1, 300, HOUGH_CIRCLES_MD)
        HC_slider.makeNewSlideObject(p1, 1, 100, HOUGH_CIRCLES_P1)
        HC_slider.makeNewSlideObject(p2, 1, 100, HOUGH_CIRCLES_P2)
        HC_slider.makeNewSlideObject(minR, 0, 100, HOUGH_CIRCLES_MINR)
        HC_slider.makeNewSlideObject(maxR, 0, 100, HOUGH_CIRCLES_MAXR)
        
    if INCLUDE_GSLIDER:
        GRAY_slider = make_slider("Gray Slider")
        GRAY_slider.makeNewSlideObject(gb, 1, 10, 5)
        GRAY_slider.makeNewSlideObject(mb, 1, 10, 5)
        GRAY_slider.makeNewSlideObject(erode, 1, 10, 3)
        GRAY_slider.makeNewSlideObject(dilate, 1, 10, 3)
    
    while cap.isOpened():
       	# Capture frame-by-frame
       	if args.input == 'cam':
            ret, frame = cap.read()
        else:
            frame = cv2.imread(input_stream)
            
        Yloc = int(y - 1.5*r)
        Xloc = int(x - 1.5*r)
        print("radius: ", r)
        print("2")
        #localize the detection of Circle for faster processing
        if(r > 0):
            localized_frame = frame[Yloc:Yloc + 3*r, Xloc:Xloc+3*r]
        else:
            localized_frame = frame;
            
        #print("localized frame size: {}".format(localized_frame.size))
        #print("3")
        
        
        # load the image, clone it for output, and then convert it to grayscale
        if localized_frame.size == 0:
            r = 0
            continue
            
        #print("4")
            
        tmp_gray = cv2.cvtColor(localized_frame, cv2.COLOR_BGR2GRAY)
        gray = tmp_gray.copy()
        
        if INCLUDE_GSLIDER:
            gb_val = GRAY_slider.getTrackBarValue(gb)
            mb_val = GRAY_slider.getTrackBarValue(mb)
            erode_val = GRAY_slider.getTrackBarValue(erode)
            dilate_val = GRAY_slider.getTrackBarValue(dilate)
            
            if(gb_val % 2 == 0):
                gb_val = gb_val-1
            if(mb_val % 2 == 0):
                mb_val = mb_val-1
        else:
            gb_val = 5
            mb_val = 5
            erode_val = 3
            dilate_val = 3
        
        output = frame.copy()
        # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
        gray = cv2.GaussianBlur(gray,(gb_val,gb_val),0);
        gray = cv2.medianBlur(gray,mb_val)
        
        # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3.5)
        
        #kernel = np.ones((3,3),np.uint8)
        gray = cv2.erode(gray,np.ones((erode_val,erode_val),np.uint8),iterations = 1)
        # gray = erosion
        
        gray = cv2.dilate(gray,np.ones((dilate_val,erode_val),np.uint8),iterations = 1)
        # gray = dilation

        # get the size of the final image
        # img_size = gray.shape
        # print img_size
        
        # read trackbar positions for all
        if INCLUDE_HCSLIDER:
            hc_dp = HC_slider.getTrackBarValue(dp)
            hc_md = HC_slider.getTrackBarValue(md)
            hc_p1 = HC_slider.getTrackBarValue(p1)
            hc_p2 = HC_slider.getTrackBarValue(p2)
            hc_minR = HC_slider.getTrackBarValue(minR)
            hc_maxR = HC_slider.getTrackBarValue(maxR)
        else:
            hc_dp = HOUGH_CIRCLES_DP = 1
            hc_md = HOUGH_CIRCLES_MD = 260
            hc_p1 = HOUGH_CIRCLES_P1 = 30
            hc_p2 = HOUGH_CIRCLES_P2 = 100
            hc_minR = HOUGH_CIRCLES_MINR = 0
            hc_maxR = HOUGH_CIRCLES_MAXR = 0
        
        # detect circles in the image
        if DEBUG_MODE:
            processTimeStart = time.time()
            
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, hc_dp, hc_md, param1=hc_p1, param2=hc_p2, minRadius=hc_minR, maxRadius=hc_maxR)
        
        if DEBUG_MODE:
            processTimeEnd = time.time()
            processTime = processTimeEnd - processTimeStart
            print("Process Time: {}".format(processTime))
        # print circles
        
        # ensure at least some circles were found
        # the detection of circle must be 1, otherwise the localized frame will depend on the last detected circle
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle in the image
                # corresponding to the center of the circle
                #cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                tmpx = Xloc + x
                tmpy = Yloc + y
                cv2.rectangle(output, (tmpx - 5, tmpy - 5), (tmpx + 5, tmpy + 5), (0, 128, 255), -1)
                cv2.circle(output, (tmpx, tmpy), r, (0, 255, 0), 4)
                #cv2.rectangle(output, (int(tmpx - 1.5*r), int(tmpy - 1.5*r)), (int(tmpx + 1.5*r), int(tmpy + 1.5*r)), (0, 128, 255), -1)
                #time.sleep(0.5)
                #print ("Column Number: {}".format(x))
                #print ("Row Number: {}".format(y))
                #print ("Radius: {}".format(r))
            
            # detect line in the image
            lines = findDial(tmp_gray)
            '''or x1,y1,x2,y2 in lines:
                cv2.line(output,(x1,y1),(x2,y2),(255,0,0),5)
            if lines is not None:
                for line in lines:
                    for x1,y1,x2,y2 in line:
                         #perimeter.append(abs(x2-x1) + abs(y2-y1))
                         cv2.line(output,(Xloc + x1,Yloc + y1),(Xloc + x2,Yloc + y2),(255,0,0),5)
                #cv2.line(output,(x1,y1),(x2,y2),(255,0,0),5)'''
            
            x = tmpx
            y = tmpy
        else:
            print("No circle found")
            y = 0
            x = 0
            r = 0

        # Display the resulting frame
        #cv2.imshow('localized frame', localized_frame)
        cv2.imshow('frame',output)
        cv2.imshow('gray', gray)
        if cv2.waitKey(1) & 0xFF == ord('q') or args.input != 'cam':
            break
        
    if args.input != 'cam':
        while(1):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    sys.exit(main() or 0)

