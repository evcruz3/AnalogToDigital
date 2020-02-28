import cv2
import sys
import global_defines as settings
import numpy as np
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
def track(img, tracker_type, circleTracker, dialTracker):
    bboxes = [(0,0,0,0),(0,0,0,0)]
    frame = img.copy()
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_lowerbound = np.array([settings.HSV_HUL, settings.HSV_SAL, settings.HSV_VAL])
    hsv_upperbound = np.array([settings.HSV_HUH, settings.HSV_SAH, settings.HSV_VAH])
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    mask = cv2.dilate(mask,np.ones((5,5),np.uint8),iterations = 1)
    
    res = cv2.bitwise_and(frame, frame, mask=mask) #filter inplace
    
    timer = cv2.getTickCount()
    # Update tracker
    ok1, bboxes[0] = circleTracker.update(frame)
    ok2, bboxes[1] = dialTracker.update(res)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    print("FPS: ", fps)
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok1 and ok2:
        # Tracking success
        
        if settings.DEBUG_MODE:
            cv2.putText(frame, "Tracking success", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
            for i, newbox in enumerate(bboxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, settings.tracker_colors[i], 2, 1)
            
        success = 1
    else :
        # Tracking failure
        if settings.DEBUG_MODE:
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
        success = 0
        
    if settings.DEBUG_MODE:
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)
        
    return bboxes, success
    
    
