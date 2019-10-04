# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 04:07:46 2019

@author: kartikay
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("matching-template.jpg")
img2 =cv2.imread("matchimg.jpg")

orb = cv2.ORB_create(nfeatures=1500)

kp1, des1 = orb.detectAndCompute(img1,None)#detects keypoints on img ,  it the array of numbers
kp2, des2 = orb.detectAndCompute(img2,None)# none = mask(no mask used here)
#descriptor defines feature independent of lightning color 

#bruteforce matching
# its going to compare the descriptors of img 1 to img2 nd returns the closest matches abck
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)

matches = bf.match(des1,des2)
matches = sorted(matches, key= lambda x:x.distance)#sorts matches by distance,from best to worst manner

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None,flags=2)



img = cv2.drawKeypoints(img2,kp2,None)
cv2.imshow("",img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
