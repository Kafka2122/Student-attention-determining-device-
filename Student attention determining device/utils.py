#from picamera.array import PiRGBArray
#from picamera import PiCamera
from picamera2 import Picamera2

import time
import cv2
from cv2 import *

def click_pic():
    
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    result, img = cam.read()
    #print(result)
    return img

def preprocess(image):
    #Initial 
    return image[290:,:]

# def load_model():
#     model =
