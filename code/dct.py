
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

#input = cv2.imread('icon.jpg')


#input?? 
def discrete_cosine_transform (imgsum, input):
    width, height, _ = input.shape
    img = np.zeros_like(input)
    
    for i in range (0, width):
        for j in range(0, height):
            if i == 0:
                xi = 1 /np.sqrt(width)
            else:
                xi = np.sqrt(2 / width)
            if j == 0:
                xj = 1 /np.sqrt(height)
            else:
                xj = np.sqrt(2 / height)
                        
            wangle = (2 * i + 1) * (i) * math.pi / (2 * width)
            hangle = (2 * j + 1) * (j) * math.pi / (2 * height)
            
            imgsum = imgsum + input[i, j] * math.cos(wangle) * math.cos(hangle)
            
    img = xi * xj * imgsum



imgt = discrete_cosine_transform (0, input)
imgplot = plt.imshow(imgt, cmap='gray')
plt.show()
                
            