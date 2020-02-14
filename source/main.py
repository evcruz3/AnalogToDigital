#!/usr/bin/env python
# import the necessary packages
import numpy as np
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import logging as log
import cv2
import time

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)

    return parser

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

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

    kernel = np.ones((3,3),np.uint8)
    
    #must find a way to localize the location for HoughCircles
    
    y = 0;
    x = 0;
    r = 0;
    
    while cap.isOpened():
       	# Capture frame-by-frame
       	if args.input == 'cam':
            ret, frame = cap.read()
        else:
            frame = cv2.imread(input_stream)
            
        Yloc = y - 2*r
        Xloc = x - 2*r

        #localize the detection of Circle for faster processing
        if(r != 0):
            localized_frame = frame[Yloc:Yloc + 4*r, Xloc:Xloc+4*r]
            cv2.imshow('localized frame', localized_frame)
        else:
            localized_frame = frame;
        
        # load the image, clone it for output, and then convert it to grayscale
                
        output = frame.copy()
        gray = cv2.cvtColor(localized_frame, cv2.COLOR_BGR2GRAY)
        
        # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
        gray = cv2.GaussianBlur(gray,(5,5),0);
        gray = cv2.medianBlur(gray,5)
        
        # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3.5)
        
        #kernel = np.ones((3,3),np.uint8)
        gray = cv2.erode(gray,kernel,iterations = 1)
        # gray = erosion
        
        gray = cv2.dilate(gray,kernel,iterations = 1)
        # gray = dilation

        # get the size of the final image
        # img_size = gray.shape
        # print img_size
        
        # detect circles in the image
        processTimeStart = time.time()
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 260, param1=30, param2=100, minRadius=10, maxRadius=0)
        processTimeEnd = time.time()
        processTime = processTimeEnd - processTimeStart
        
        print(processTime)
        # print circles
        
        # ensure at least some circles were found
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
                #time.sleep(0.5)
                #print ("Column Number: {}".format(x))
                #print ("Row Number: {}".format(y))
                #print ("Radius: {}".format(r))
            
            x = Xloc + x
            y = Yloc + y
        else:
            y = 0
            x = 0
            r = 0

        

        # Display the resulting frame
        #cv2.imshow('gray',gray)
        cv2.imshow('frame',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    sys.exit(main() or 0)

