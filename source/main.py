#!/usr/bin/env python
# import the necessary packages
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import logging as log
import time
import circle_detector as cdetect
import global_defines as settings
import cv2
import slider
import numpy as np

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
    gray = cv2.GaussianBlur(tmp_gray,(settings.GAUSSIAN_BLUR_KSIZE, settings.GAUSSIAN_BLUR_KSIZE),0)
    
    ret, gray = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    
    #kernel = np.ones((3,3),np.uint8)
    #gray = cv2.dilate(gray,kernel,iterations = 3)
    
    edges = cv2.Canny(gray, settings.CANNY_LOW_THRESHOLD, settings.CANNY_HIGH_THRESHOLD)
    
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    
    lines = cv2.HoughLinesP (edges, 
                            settings.HOUGH_LINESP_RHO, 
                            settings.HOUGH_LINESP_THETA, 
                            settings.HOUGH_LINESP_THRESHOLD, 
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
    settings.init()

    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    if(args.debug):
        settings.DEBUG_MODE = 1
    if(args.hcslider):
        settings.INCLUDE_HCSLIDER = 1
    if(args.gslider):
        settings.INCLUDE_GSLIDER = 1

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

    # Previous circle's coordinates will be used to localize frame for the next circle detection
    y = 0;
    x = 0;
    r = 0;
    #n = 0;
    
    if settings.INCLUDE_HCSLIDER:
        HC_slider = slider.make_slider('Hough Circles Slider')
        HC_slider.makeNewSlideObject(settings.dp, 1, 5, settings.HOUGH_CIRCLES_DP)
        HC_slider.makeNewSlideObject(settings.md, 1, 300, settings.HOUGH_CIRCLES_MD)
        HC_slider.makeNewSlideObject(settings.p1, 1, 100, settings.HOUGH_CIRCLES_P1)
        HC_slider.makeNewSlideObject(settings.p2, 1, 100, settings.HOUGH_CIRCLES_P2)
        HC_slider.makeNewSlideObject(settings.minR, 0, 100, settings.HOUGH_CIRCLES_MINR)
        HC_slider.makeNewSlideObject(settings.maxR, 0, 100, settings.HOUGH_CIRCLES_MAXR)
        
    if settings.INCLUDE_GSLIDER:
        GRAY_slider = slider.make_slider("Gray Slider")
        GRAY_slider.makeNewSlideObject(settings.gb, 1, 10, 5)
        GRAY_slider.makeNewSlideObject(settings.mb, 1, 10, 5)
        GRAY_slider.makeNewSlideObject(settings.erode, 1, 10, 3)
        GRAY_slider.makeNewSlideObject(settings.dilate, 1, 10, 3)
    
    while cap.isOpened():
       	# Capture frame-by-frame
        print("Debug Mode Main: ", settings.DEBUG_MODE)
       	if args.input == 'cam':
            ret, frame = cap.read()
        else:
            frame = cv2.imread(input_stream)
            
        Yloc = int(y - 1.5*r)
        Xloc = int(x - 1.5*r)
        
        #localize the detection of Circle for faster processing
        if(r > 0):
            localized_frame = frame[Yloc:Yloc + 3*r, Xloc:Xloc+3*r]
        else:
            localized_frame = frame;
        
        # if localized frame is zero (implies failure to previously detect a circle or the circle detected is almost out of frame)
        if localized_frame.size == 0:
            r = 0
            continue
            
        output = frame.copy()
            
        # load the image, clone it for output, and then convert it to grayscale
        tmp_gray = cv2.cvtColor(localized_frame, cv2.COLOR_BGR2GRAY)
        gray = tmp_gray.copy()
        
        # Gray processing
        if settings.INCLUDE_GSLIDER:
            gray = cdetect.grayProcess(gray, GRAY_slider)
        else:
            gray = cdetect.grayProcess(gray)
        
        # detect circles in the image
        if settings.INCLUDE_HCSLIDER:
            circles = cdetect.detectCircles(gray, HC_slider)
        else:
            circles = cdetect.detectCircles(gray)

        # ensure at least some circles were found
        # the detection of circle must be 1, otherwise the localized frame will depend on the last detected circle
        # TODO: Create a handling when multiple circles are detected (Which circle shall be used?)
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
            
            # detect line in the image
            # TODO: Either detect through line or through HSV (depending on the bus dashboard model)
            lines = findDial(tmp_gray)
            
            x = tmpx
            y = tmpy
        else:
            if settings.DEBUG_MODE:
                print("No circle found")
            y = 0
            x = 0
            r = 0

        # Display the resulting frame
        #cv2.imshow('localized frame', localized_frame)
        cv2.imshow('frame',output)
        cv2.imshow('gray', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    '''if args.input != 'cam':
        while(1):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    sys.exit(main() or 0)

