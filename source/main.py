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
import tracker

#TODO: Handling of multiple detected circles
#TODO: Dial detection through HSV and Line Detection (ok)
#TODO: Conversion of angle to data (ok)
#TODO: Assess angle resolution (should it be int, double, float, etc)
#TODO: Setup args for angles and range values (ok)
#TODO: Make the calibration more customer-centric
#TODO: Add tracking for faster video processing

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-minA", "--min_angle", type=int, required=True, default=211, help='Specify the angle where the zero value is')
    args.add_argument("-maxA", "--max_angle", type=int, required=True, default=0, help='Specify the angle where the maximum value is')
    args.add_argument("-maxV", "--max_value", type=int, required=True, default=180, help='Specifiy the maximum value')
    args.add_argument("-u", "--unit", type=str, default='kph', help='Specify the units')
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
    if settings.INCLUDE_HSVSLIDER:
        cv2.imshow('HSV', mask)
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
    
def convertAngleToData(angle):
    if(angle <= 270):
        transformedAngle = settings.MIN_ANGLE - angle
    else:
        transformedAngle = settings.MIN_ANGLE + (360-angle)
        
    data = (transformedAngle/settings.ANGLE_RANGE)*settings.MAX_VALUE
    if settings.DEBUG_MODE:
        log.info("[{0}] {1} {2}".format(time.time(), data, settings.UNIT))
        
    return data
    
    
def process_args(args):
    if(args.debug):
        settings.DEBUG_MODE = 1
        log.info("Debug Mode is enabled")
    if(args.hcslider):
        settings.INCLUDE_HCSLIDER = 1
        log.info("Hough Circles Slider is enabled")
    if(args.gslider):
        settings.INCLUDE_GSLIDER = 1
        log.info("Gray Slider is enabled")
    if(args.hsvslider):
        settings.INCLUDE_HSVSLIDER = 1
        log.info("HSV Slider is enabled")

    if args.input == 'cam':
        settings.input_stream = '/dev/video0'
    else:
        settings.input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
        
    settings.UNIT = args.unit
    settings.MIN_ANGLE = args.min_angle
    settings.MAX_ANGLE = args.max_angle
    settings.MAX_VALUE = args.max_value
    
    if(args.max_angle <= 270):
        settings.ANGLE_RANGE = args.min_angle - args.max_angle
    else:
        print("TEST")
        settings.ANGLE_RANGE = args.min_angle + (360 - args.max_angle)
    
    log.info("Minimum: 0 at {} deg".format(settings.MIN_ANGLE))
    log.info("Maximum: {0} at {1} deg".format(settings.MAX_VALUE, settings.MAX_ANGLE))
    log.info("Angle Range: {} deg".format(settings.ANGLE_RANGE))
    
    
def getFrame(cap):
    if settings.input_stream == '/dev/video0':
        _, frame = cap.read()
    else:
        frame = cv2.imread(settings.input_stream)
        '''width = 640
        height = int(frame.shape[0] * width/frame.shape[1])
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)'''
        
    return frame

def sampleData(circleLocation, dialLocation, frame):
    tmpx = circleLocation[0]
    tmpy = circleLocation[1]
    dial_x = dialLocation[0]
    dial_y = dialLocation[1]
    output = frame.copy()
    
    hypotenuse = distance(tmpx, tmpy, dial_x, dial_y)
    horizontal = math.fabs(tmpx-dial_x)#distance(tmpx, 0, dial_x, 0)
    vertical = math.fabs(tmpy-dial_y)#distance(0, tmpy, 0, dial_y)
    angle = np.arcsin(vertical/hypotenuse)*180.0/math.pi
    
    if dial_y < tmpy and dial_x > tmpx:
        final_angle = int(angle)
    elif dial_y < tmpy and dial_x < tmpx:
        final_angle = int(180 - angle)
    elif dial_y > tmpy and dial_x < tmpx:
        final_angle = int(180 + angle)
    elif dial_y > tmpy and dial_x > tmpx:
        final_angle = int(360 - angle)
    
    if settings.DEBUG_MODE:
        cv2.line(output, (tmpx, tmpy), (dial_x, dial_y), (0, 0, 255), 2)
        cv2.line(output, (tmpx, tmpy), (dial_x, tmpy), (0, 0, 255), 2)
        cv2.line(output, (dial_x, dial_y), (dial_x, tmpy), (0, 0, 255), 2)
        
        #put angle text (allow for calculations upto 180 degrees)
        angle_text = str(final_angle)
        
        #CHANGE FONT HERE
        cv2.putText(output, angle_text, (tmpx-30, tmpy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 128, 229), 2)
    
        #cv2.imshow('sampleData', output)
    data = convertAngleToData(final_angle) 
    
    return data

def findCircleAndDial(frame, Cx = 0, Cy = 0, Cr = 0, HC_slider = None, GRAY_slider = None, HSV_slider = None):
    Yloc = int(Cy - 1.5*Cr)
    Xloc = int(Cx - 1.5*Cr)
    
    dial_x, dial_y = (0, 0)
    
    #localize the detection of Circle for faster processing
    if(Cr > 0):
        localized_frame = frame[Yloc:Yloc + 3*Cr, Xloc:Xloc+3*Cr]
    else:
        localized_frame = frame;
    
    # if localized frame is zero (implies failure to previously detect a circle or the circle detected is almost out of frame)
    if localized_frame.size == 0:
        Cr = 0
        return (0,0,0), (0,0), 0, 0, 0
        
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
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # loop over the (x, y) coordinates and radius of the circles
        for (Cx, Cy, Cr) in circles:
            # draw the circle in the output image, then draw a rectangle in the image
            # corresponding to the center of the circle
            #cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            tmpx = Xloc + Cx
            tmpy = Yloc + Cy
            if settings.DEBUG_MODE:
                cv2.rectangle(output, (tmpx - 5, tmpy - 5), (tmpx + 5, tmpy + 5), (0, 128, 255), -1)
                cv2.circle(output, (tmpx, tmpy), Cr, (0, 255, 0), 4)
        
                #cv2.imshow('find circle', output)
        # detect dial in the image
        if settings.INCLUDE_HSVSLIDER:
            (dial_x, dial_y), found_dial = findDial(localized_frame, HSV_slider)
        else:
            (dial_x, dial_y), found_dial = findDial(localized_frame)
        
        if found_dial:
            locationSuccess = 1 
            if settings.DEBUG_MODE:
                log.info("Dial found")
            
        else:
            if settings.DEBUG_MODE:
                log.info("No dial found")
            locationSuccess = 0
        #---------------------------------------------------------------------
        
        Cx = tmpx
        Cy = tmpy
    else:
        if settings.DEBUG_MODE:
            log.info("No Circle found")
        Cy = 0
        Cx = 0
        Cr = 0
        locationSuccess = 0
        
    circleLocation = [Cx, Cy, Cr]
    dialLocation = [dial_x, dial_y]
        
    return circleLocation, dialLocation, locationSuccess
    

def main():
    settings.init()
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    process_args(args)
    
    cap = cv2.VideoCapture(settings.input_stream) # Set Capture Device, in case of a USB Webcam try 1, or give -1 to get a list of available devices

    #Set Width and Height 
    # cap.set(3,1280)
    # cap.set(4,720)

    # The above step is to set the Resolution of the Video. The default is 640x480.
    # This example works with a Resolution of 640x480.
    
    
    HC_slider = None
    if settings.INCLUDE_HCSLIDER:
        HC_slider = slider.make_slider('Hough Circles Slider')
        HC_slider.makeNewSlideObject(settings.dp,   1, 5,   settings.HOUGH_CIRCLES_DP)
        HC_slider.makeNewSlideObject(settings.md,   1, 300, settings.HOUGH_CIRCLES_MD)
        HC_slider.makeNewSlideObject(settings.p1,   1, 100, settings.HOUGH_CIRCLES_P1)
        HC_slider.makeNewSlideObject(settings.p2,   1, 100, settings.HOUGH_CIRCLES_P2)
        HC_slider.makeNewSlideObject(settings.minR, 0, 100, settings.HOUGH_CIRCLES_MINR)
        HC_slider.makeNewSlideObject(settings.maxR, 0, 100, settings.HOUGH_CIRCLES_MAXR)
        
    GRAY_slider = None
    if settings.INCLUDE_GSLIDER:
        GRAY_slider = slider.make_slider("Gray Slider")
        GRAY_slider.makeNewSlideObject(settings.gb,     1, 10, 5)
        GRAY_slider.makeNewSlideObject(settings.mb,     1, 10, 5)
        GRAY_slider.makeNewSlideObject(settings.erode,  1, 10, 3)
        GRAY_slider.makeNewSlideObject(settings.dilate, 1, 10, 3)
        
    HSV_slider = None
    if settings.INCLUDE_HSVSLIDER:
        HSV_slider = slider.make_slider("HSV Slider")
        HSV_slider.makeNewSlideObject(settings.hl, 0, 179, settings.HSV_HUL)
        HSV_slider.makeNewSlideObject(settings.hh, 0, 179, settings.HSV_HUH)
        HSV_slider.makeNewSlideObject(settings.sl, 0, 255, settings.HSV_SAL)
        HSV_slider.makeNewSlideObject(settings.sh, 0, 255, settings.HSV_SAH)
        HSV_slider.makeNewSlideObject(settings.vl, 0, 255, settings.HSV_VAL)
        HSV_slider.makeNewSlideObject(settings.vh, 0, 255, settings.HSV_VAH)
    
    circleLocation = (0,0,0) #X, Y, R
    dialLocation = (0,0)
    
    #pseudocode
    '''
    while(cap.isOpened())
        getFrame (ok)
        detectCircleAndDial
        if detect_success:
            sampleData
            while(1)
                getFrame
                track
                if tracking_success
                    sampleData
                else
                    break
    '''
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
    if tracker_type == 'BOOSTING':
        tracker1 = cv2.TrackerBoosting_create()
        tracker2 = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker1 = cv2.TrackerMIL_create()
        tracker2 = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker1 = cv2.TrackerKCF_create()
        tracker2 = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker1 = cv2.TrackerTLD_create()
        tracker2 = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker1 = cv2.TrackerMedianFlow_create()
        tracker2 = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker1 = cv2.TrackerGOTURN_create()
        tracker2 = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker1 = cv2.TrackerMOSSE_create()
        tracker2 = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker1 = cv2.TrackerCSRT_create()
        tracker2 = cv2.TrackerCSRT_create()
    
    while cap.isOpened():
       	# Capture frame-by-frame
       	
        frame = getFrame(cap)
        circleLocation, dialLocation, locationSuccess = findCircleAndDial(frame, circleLocation[0], circleLocation[1], circleLocation[2], HC_slider, GRAY_slider, HSV_slider)
        
        '''if settings.DEBUG_MODE:
            cv2.imshow('output',output)
        if settings.INCLUDE_GSLIDER:
            cv2.imshow('gray', gray)'''
            
        if locationSuccess:
            log.info("Circle and Dial Detection Successful")
            data = sampleData(circleLocation, dialLocation, frame)
            while (1):
                frame = getFrame(cap)
                #while(1):
                #    cv2.imshow('output',output)
                bbox1 = (circleLocation[0] - 5, circleLocation[1] - 5, 10, 10)
                bbox2 = (dialLocation[0] - 5, dialLocation[1] - 5, 10, 10)
                tracker1.init(frame, bbox1)
                tracker2.init(frame, bbox2)
                bbox1, bbox2, track_success, output = tracker.track(frame, tracker_type, tracker1, bbox1, tracker2, bbox2)
                if track_success:
                    log.info("Track success")
                    circleLocation = (int(bbox1[0] + 5), int(bbox1[1] + 5), circleLocation[2])
                    dialLocation = (int(bbox2[0] + 5), int(bbox2[1] + 5))
                    data = sampleData(circleLocation, dialLocation, frame)
                else:
                    break
                    
                #if settings.DEBUG_MODE:
                #    cv2.imshow('frame',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    sys.exit(main() or 0)

