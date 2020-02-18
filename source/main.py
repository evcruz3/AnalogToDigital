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
import math

#TODO: Dial detection through HSV and Line Detection (ok)

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
    args.add_argument("-hs", "--hsvslider", action='store_true', default=False, help='Enables the hsv slider')

    return parser

def findDial(frame, HSV_slider = None):
    # convert to HSV from BGR
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # read trackbar positions for all
    if HSV_slider is not None:
        hul = HSV_slider.getTrackBarValue(settings.hl)
        huh = HSV_slider.getTrackBarValue(settings.hh)
        sal = HSV_slider.getTrackBarValue(settings.sl)
        sah = HSV_slider.getTrackBarValue(settings.sh)
        val = HSV_slider.getTrackBarValue(settings.vl)
        vah = HSV_slider.getTrackBarValue(settings.vh)
    else:
        hul = settings.HSV_HUL
        huh = settings.HSV_HUH
        sal = settings.HSV_SAL
        sah = settings.HSV_SAH
        val = settings.HSV_VAL
        vah = settings.HSV_VAH

    hsv_lowerbound = np.array([hul, sal, val])
    hsv_upperbound = np.array([huh, sah, vah])
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    res = cv2.bitwise_and(frame, frame, mask=mask) #filter inplace
    cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxcontour = max(cnts, key=cv2.contourArea)

        #Find center of the contour 
        M = cv2.moments(maxcontour)
        if M['m00'] > 0 and cv2.contourArea(maxcontour) > 1000:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return (cx, cy), True
        else:
            return (700, 700), False #faraway point
    else:
        return (700, 700), False #faraway point
    
    
def distance(x1, y1, x2, y2):
    dist = math.sqrt(math.fabs(x2-x1)**2 + math.fabs(y2-y1)**2)
    return dist

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
    if(args.hsvslider):
        settings.INCLUDE_HSVSLIDER = 1

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
        
    if settings.INCLUDE_HSVSLIDER:
        HSV_slider = slider.make_slider("HSV Slider")
        HSV_slider.makeNewSlideObject(settings.hl, 0, 179, settings.HSV_HUL)
        HSV_slider.makeNewSlideObject(settings.hh, 0, 179, settings.HSV_HUH)
        HSV_slider.makeNewSlideObject(settings.sl, 0, 255, settings.HSV_SAL)
        HSV_slider.makeNewSlideObject(settings.sh, 0, 255, settings.HSV_SAH)
        HSV_slider.makeNewSlideObject(settings.vl, 0, 255, settings.HSV_VAL)
        HSV_slider.makeNewSlideObject(settings.vh, 0, 255, settings.HSV_VAH)
    
    while cap.isOpened():
       	# Capture frame-by-frame
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
            
            # detect dial in the image
            if settings.INCLUDE_HSVSLIDER:
                (dial_x, dial_y), found_dial = findDial(localized_frame, HSV_slider)
            else:
                (dial_x, dial_y), found_dial = findDial(localized_frame)
            
            if found_dial:
                hypotenuse = distance(tmpx, tmpy, dial_x, dial_y)
                horizontal = distance(tmpx, tmpy, dial_x, dial_y)
                vertical = distance(tmpx, tmpy, dial_x, dial_y)
                angle = np.arcsin(vertical/hypotenuse)*180.0/math.pi
                
            cv2.line(output, (tmpx, tmpy), (dial_x, dial_y), (0, 0, 255), 2)
            cv2.line(output, (tmpx, tmpy), (dial_x, tmpy), (0, 0, 255), 2)
            cv2.line(output, (dial_x, dial_y), (dial_x, tmpy), (0, 0, 255), 2)
            
            #put angle text (allow for calculations upto 180 degrees)
            angle_text = ""
            if dial_y < tmpy and dial_x > tmpx:
                angle_text = str(int(angle))
            elif dial_y < tmpy and dial_x < tmpx:
                angle_text = str(int(180 - angle))
            elif dial_y > tmpy and dial_x < tmpx:
                angle_text = str(int(180 + angle))
            elif dial_y > tmpy and dial_x > tmpx:
                angle_text = str(int(360 - angle))
            
            #CHANGE FONT HERE
            cv2.putText(output, angle_text, (tmpx-30, tmpy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 128, 229), 2)
            
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
        #cv2.imshow('gray', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    sys.exit(main() or 0)

