# sudo apt-get update
# sudo apt-get upgrade
# mkdir Python
# mkdir tLite
# git clone https://github.com/tensorflow/examples --depth 1
# cd examples
# cd lite/examples/object_detection/raspberry_pi
# sh setup.sh
# mv efficientdet_lite0.tflite ~/Python
# mv utils.py ~/Python

import cv2
import time
from picamera2 import Picamera2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

model='efficientdet_lite0.tflite'
num_threads =4

dispW=1280
dispH=720

picam2 = Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.preview_configuration.main.format='RGB888'
picam2.preview_configuration.align()
picam2.configure('preview')
picam2.start()

webCam="/dev/video2"
cam = cv2.VideoCapture(webCam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispH)
cam.set(cv2.CAP_PROP_FPS,30)

pos=(20,60)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 1.5
weight = 3
myColor =(255,0,0)

fps=0

base_options=core.BaseOptions(file_name=model,use_coral=False,num_threads=num_threads)
detection_options=processor.DetectionOptions(max_results=8,score_threshold=0.3)
options = vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)
tStart=time.time()

while True:
    # ret, im =cam.read()
    im=picam2.capture_array()
    im=cv2.flip(im,-1)
    imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    imTensor=vision.TensorImage.create_from_array(imRGB)
    detections=detector.detect(imTensor)
    image = utils.visualize(im,detections)
    cv2.putText(im,str(int(fps))+" FPS",pos,font,height,myColor,weight)
    cv2.imshow('Camera',im)
    if cv2.waitKey(1)==ord('q'):
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps=.9*fps+0.1*1/loopTime
    # print(fps)
    tStart=time.time()
cv2.destroyAllWindows