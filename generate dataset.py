# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:29:51 2020

@author: Arnob Chowdhury
"""
# 1.open original image
# 2.convert to grayscale
# 3.Contour Detection
# 4.Get Areas of Largest Contours
# --> For an Infected cell we may get Many Countours and for uninfected cell 
# we may get only one countour
import cv2
import os
import numpy as np
import csv
import glob

label = "Uninfected"

"""im = cv2.imread("cell_images\\Uninfected\\C1_thinF_IMG_20150604_104722_cell_9.png", cv2.IMREAD_UNCHANGED)
print(im)
cv2.imshow("image",im)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

dirList = glob.glob("cell_images\\"+label+"\\*.png")
file = open("csv/dataset.csv","a")

for img_path in dirList:
    im = cv2.imread(img_path,cv2.IMREAD_COLOR)
    im = cv2.GaussianBlur(im,(5,5),2)
    
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(im_gray,127,255,0)
    contours, _ = cv2.findContours(thresh,1,2)
    
    file.write(label)
    file.write(",")
    
    for i in range(5):
        try:
            area = cv2.contourArea(contours[i])
            file.write(str(area))
        except:
            file.write("0")
            
        file.write(",")
    file.write("\n")

