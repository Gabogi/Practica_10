import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
   
image = cv2.imread('SoloLeveling.jpg') 
   
mask = np.zeros(image.shape[:2], np.uint8) 
   
backgroundModel = np.zeros((1, 65), np.float64) 
foregroundModel = np.zeros((1, 65), np.float64) 
   
rectangle = (20, 10, 1200, 675) 
   
cv2.grabCut(image, mask, rectangle,   
            backgroundModel, foregroundModel, 
            3, cv2.GC_INIT_WITH_RECT) 
   
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
   
image = image * mask2[:, :, np.newaxis] 
   
plt.imshow(image) 
plt.colorbar()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10) 
corners = np.int0(corners) 
  
for i in corners: 
    x, y = i.ravel() 
    cv2.circle(image,(x, y), 3, 255, -1) 
  
plt.imshow(image) 
plt.show()