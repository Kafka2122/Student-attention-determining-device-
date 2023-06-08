import RPi.GPIO as GPIO
import time
import os, sys
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
#setup output pins
GPIO.setup(33, GPIO.OUT)      #//GPIO13#l
GPIO.setup(31, GPIO.OUT)      #//GPIO06#m
GPIO.setup(36, GPIO.OUT)      #//GPIO16#h
GPIO.setup(38, GPIO.OUT)      #//GPIO20
GPIO.setup(40, GPIO.OUT)      #//GPIO21
GPIO.setup(35, GPIO.OUT)      #//GPIO19
GPIO.setup(37, GPIO.OUT)      #//GPIO26

#define 7 segment digits
digitclr=[1,1,1,1,1,1,1]

low =[0,1,1,1,1,1,1]
mid =[0,0,1,1,1,1,1]
high=[0,0,0,1,1,1,1]
gpin=[33,31,36,38,40,35,37]
#routine to clear and then write to display


def disp_low():
    digitclr=[1,1,1,1,1,1,1]
    low =[0,0,0,0,1,0,0]
    for x in range (0,7):
        GPIO.output(gpin[x], digitclr[x])
    for x in range (0,7):
        GPIO.output(gpin[x], low[x])
    time.sleep(1)

def disp_mid():
    digitclr=[1,1,1,1,1,1,1]
    mid =[0,0,0,0,1,0,1]
    for x in range (0,7):
        GPIO.output(gpin[x], digitclr[x])
    for x in range (0,7):
        GPIO.output(gpin[x], mid[x])
    time.sleep(1)

def disp_high():
    digitclr=[1,1,1,1,1,1,1]
    high=[1,0,0,0,1,0,1]
    for x in range (0,7):
        GPIO.output(gpin[x], digitclr[x])
    for x in range (0,7):
        GPIO.output(gpin[x], high[x])
    time.sleep(1)


if __name__=='__main__':
    disp_low()
    disp_mid()
    disp_high()
    for x in range(0,7):
        GPIO.output(gpin[x],1)
    time.sleep(1)
