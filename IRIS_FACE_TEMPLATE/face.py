# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:48:57 2021

@author: OKOK PROJECTS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

    
import sklearn.datasets as datasets

import seaborn as sns
# Reading the original image
img = plt.imread('elon.jpg')

plt.imshow(img)
print("",img.shape)
x = 150
y = 40
h = 330
w = 270

face = img[y:y+h, x:x+w]

plt.imshow(face)

# Saving the cropped image 
cv2.imwrite('E:/IRIS/elon_face.jpg',face)

# Reading original and cropped image
elon = plt.imread('elon.jpg')
elon_face = plt.imread('elon_face.jpg')

elon.shape

elon_face.shape

height,width,channels = elon_face.shape

height

width

# Different methods used in template matching 
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


for m in methods:
    
    # Create a copy of the image
    original = elon.copy()
    
    # Get the actual function instead of the string
    method = eval(m)

    # Apply template Matching with the method
    res = cv2.matchTemplate(original,elon_face,method)
    
    # Grab the Max and Min values, their locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # Assigning the top left of the rectangle
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc    
    else:
        top_left = max_loc
        
    # Assign the Bottom Right of the rectangle
    bottom_right = (top_left[0] + width, top_left[1] + height)

    # Draw the Red Rectangle
    cv2.rectangle(original,top_left, bottom_right, 255, 10)

    # Plot the Images
    plt.subplot(121) #plt.subplot(row col select)
    plt.imshow(res)
    plt.title('Result of Template Matching')
    
    plt.subplot(122)
    plt.imshow(original)
    plt.title('Detected Point for face')
    plt.suptitle(m)
    
    
    plt.show()
    print('\n')
    print('\n')
    
t3=pd.read_csv("face.csv")
t3.head()
t3.info()
t3.describe().T
print(t3["template"].value_counts()) 
sns.countplot(x="template", data=t3)
plt.show()

sns.swarmplot(x="template", y="value1", data=t3)
plt.show()
sns.swarmplot(x="template", y="value2", data=t3)
plt.show()
sns.swarmplot(x="template", y="value3", data=t3)
plt.show()
sns.swarmplot(x="template", y="value4", data=t3)
plt.show()
