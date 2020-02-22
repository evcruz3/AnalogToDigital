import slider
import global_defines as settings
import numpy as np
import cv2

def grayProcess(gray_image, GRAY_slider = None):
    global DEBUG_MODE
    
    if GRAY_slider is not None:
        gb_val = GRAY_slider.getTrackBarValue(settings.gb)
        mb_val = GRAY_slider.getTrackBarValue(settings.mb)
        erode_val = GRAY_slider.getTrackBarValue(settings.erode)
        dilate_val = GRAY_slider.getTrackBarValue(settings.dilate)
        
        if(gb_val % 2 == 0):
            gb_val = gb_val-1
        if(mb_val % 2 == 0):
            mb_val = mb_val-1
    else:
        gb_val = settings.GAUSSIAN_BLUR_KSIZE
        mb_val = settings.MEDIAN_BLUR_KSIZE
        erode_val = settings.ERODE_KERNEL
        dilate_val = settings.DILATE_KERNEL

    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray = cv2.GaussianBlur(gray_image,(gb_val,gb_val),0);
    gray = cv2.medianBlur(gray,mb_val)
    
    # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3.5)
    
    gray = cv2.erode(gray,np.ones((erode_val,erode_val),np.uint8),iterations = 1)
    # gray = erosion
    
    gray = cv2.dilate(gray,np.ones((dilate_val,erode_val),np.uint8),iterations = 1)
    # gray = dilation
    
    return gray

def detectCircles(gray, HC_slider = None):
    if HC_slider is not None:
	    hc_dp = HC_slider.getTrackBarValue(settings.dp)
	    hc_md = HC_slider.getTrackBarValue(settings.md)
	    hc_p1 = HC_slider.getTrackBarValue(settings.p1)
	    hc_p2 = HC_slider.getTrackBarValue(settings.p2)
	    hc_minR = HC_slider.getTrackBarValue(settings.minR)
	    hc_maxR = HC_slider.getTrackBarValue(settings.maxR)
    else:
        hc_dp = settings.HOUGH_CIRCLES_DP
        hc_md = settings.HOUGH_CIRCLES_MD
        hc_p1 = settings.HOUGH_CIRCLES_P1
        hc_p2 = settings.HOUGH_CIRCLES_P2
        hc_minR = settings.HOUGH_CIRCLES_MINR
        hc_maxR = settings.HOUGH_CIRCLES_MAXR
        
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, hc_dp, hc_md, param1=hc_p1, param2=hc_p2, minRadius=hc_minR, maxRadius=hc_maxR)
    
    return circles
    
    
    
