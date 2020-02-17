import cv2

def nothing(x):
    pass

class sliderObject(object):
    name = ''
    minval = 0
    maxval = 0
    defval = 0
    
    def __init__(self, name, minval, maxval,defval):
        self.name = name
        self.minval = minval
        self.maxval = maxval
        self.defval = defval

class Slider(object):
    sliderObjects = []; #array of objects
    windowName = ''
    
    def __init__(self, windowName):
        self.windowName = windowName
        self.sliderObjects = []
        cv2.namedWindow(self.windowName, flags = cv2.WINDOW_AUTOSIZE)
        
    def makeNewSlideObject(self, name='Slider', minval=0, maxval=0, defval=0):
        self.sliderObjects.append(sliderObject(name, minval, maxval, defval))
        cv2.createTrackbar(name, self.windowName, minval, maxval, nothing)
        cv2.setTrackbarPos(name, self.windowName, defval)
        
    def getTrackBarValue(self, objectName):
        value = cv2.getTrackbarPos(objectName, self.windowName)
        return value
    
        
def make_slider(windowName):
    newSlider = Slider(windowName)
    return newSlider
