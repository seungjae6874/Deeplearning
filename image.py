
import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
print ("PACKAGES LOADED")
'''
#Print Current Folder
cwd = os.getcwd()
print("Current folder is %s" %(cwd))


#print Function
def print_typeshape(img):
    print("Type is %s" %(type(img)))
    print("Shape is %s" %(img.shape,))

#Load an image
cat = imread("./cat.jpg")
print_typeshape(cat)

#Plot loaded image
plt.figure(0)
plt.imshow(cat)
plt.title("ORIGINAL CAT")
plt.show()

#load +cast to float?
cat2 = imread("./cat.jpg").astype(np.float)
print_typeshape(cat2)
#plot
plt.figure(0)
plt.imshow(cat2)
plt.title("Original image with imread.astype(np.float)")
plt.show()

#Load
cat3 = imread("./cat.jpg").astype(np.float)
print_typeshape(cat3)
#plot
plt.figure(0)
plt.imshow(cat3/255.)
plt.title("Original image with (np.float/255.")
plt.show()

#resize
catsmall = imresize(cat, [100,100,3])
print_typeshape(catsmall)
#plot
plt.figure(0)
plt.imshow(catsmall)
plt.title("Resize cat small")
plt.show()


#Grayscale

def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3], [0.299 , 0.587, 0.114])
    else:
        print("Current image is already gray!")
        return rgb


catsmallgray = rgb2gray(catsmall)
print("size of catsmall is %s" % (catsmallgray.shape,))
print("type of catsmallgray is", type(catsmallgray))

plt.imshow(catsmallgray, cmap=plt.get_cmap("gray"))
plt.title("[imshow] Gray image")
plt.colorbar()
plt.show()
'''

#see what inside in this folder
cwd = os.getcwd()
path = cwd
flist = os.listdir(path)
print("[%d] FILE ARE IN [%s]" %(len(flist),path))

for i,f in enumerate(flist):
    print("[%d] The file is [%s]" %(i,f))


valid_exts = [".jpg",".gif",".png",".jpeg"]
imgs = []
names = []
for f in flist:
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_exts:
        continue
    fullpath = os.path.join(path,f)
    imgs.append(imread(fullpath))
    names.append(os.path.splitext(f)[0])
    

for img,name in zip(imgs,names):
    plt.imshow(img)
    plt.title(name)
    plt.show()
    
