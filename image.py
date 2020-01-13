
import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
print ("PACKAGES LOADED")

#Print Current Folder
cwd = os.getcwd()
print("Current folder is %s" %(cwd))


#print Function
def print_typeshape(img):
    print("Type is %s" %(type(img)))
    print("Shape is %s" %(img.shape,))

#Load an image
cat = imread("./mygitML/cat.jpg")
print_typeshape(cat)

#Plot loaded image
plt.figure(0)
plt.imshow(cat)
plt.title("ORIGINAL CAT")
plt.show()

#load +cast to float?
cat2 = imread("./mygitML/cat.jpg").astype(np.float)
print_typeshape(cat2)
#plot
plt.figure(0)
plt.imshow(cat2)
plt.title("Original image with imread.astype(np.float)")
plt.show()
