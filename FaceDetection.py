from cv2 import cv2
#import RPi.GPIO as GPIO
#import time
#---------------------------------#
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(11,GPIO.OUT)
#GPIO.setup(9,GPIO.OUT)
#servox = GPIO.PWM(11,50)
#servoy = GPIO.PWM(9,50)
#servox.start(0)
#servoy.start(0)
#---------------------------------#
#def Rot(dutx,duty):
#    servox.ChangeDutyCycle(dutx)
#    servoy.ChangeDutyCycle(duty)
#---------------------------------#
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
while True:
    successful_frame_Read, frame = webcam.read()
    greyscaledimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    face_coords = trained_face_data.detectMultiScale(greyscaledimg)

    for (x,y,w,h) in face_coords:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (220,220,0),10)
        print(x+w/2,y-h/2)
        #print(w,h)
        #Rot(x,y)
        
    

    cv2.imshow('window',frame)
    cv2.waitKey(1)
print("fine")


