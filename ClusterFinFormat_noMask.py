import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from skimage import measure, color, io
import pickle

base_img = r"C:/Users/sammy/Downloads/SEM/SEM/Images/QiSmall1.tif"
bs = cv2.imread(base_img, 0)
print(bs.shape)
print(np.max(bs))
print(np.min(bs))
trim_limit = 884 #img1= 884, img2=884 
bs = bs[0:trim_limit, :]
plt.imshow(bs)
plt.show()

kernel = np.ones((5,5),np.float32)/25
avgd = cv2.filter2D(bs,-1,kernel)

sobelx = cv2.Sobel(avgd,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(avgd,cv2.CV_64F,0,1,ksize=5)

gradsize = sobelx**2 + sobely**2
gradsize = np.round(255*gradsize/np.max(gradsize)).astype(int)

avgd = np.round(avgd).astype(int)
plt.imshow(avgd)
plt.show()

plt.imshow(gradsize)
plt.show()


parts = 7
partition = 4
l = bs.shape[0]//parts
h = bs.shape[1]//parts
print(l)
print(h)
testIndex = 0
for i in range(0, parts**2):
    print(i)
    a = i//parts
    b = i%parts
    ind1 = a*l
    ind2 = (a+1)*l
    ind3 = b*h
    ind4 = (b+1)*h
    imarray = bs[ind1:ind2, ind3:ind4]
    imarray2 = avgd[ind1:ind2, ind3:ind4]
    imarray3 = gradsize[ind1:ind2, ind3:ind4]
    #plt.imshow(imarray)
    #plt.imshow(IDarray)
    tempImg = imarray
    tempImg2 = imarray2
    tempImg3 = imarray3
    # fig,ax = plt.subplots(1)
    # plt.imshow(masks[-1])
    # # Create a Rectangle patch
    # box = boxes[-1]
    # rect = patches.Rectangle((box[0],box[1]),(box[2]-box[0]),(box[3] - box[1]),linewidth=1,edgecolor='r',facecolor='none')
    
        # # Add the patch to the Axes
        # ax.add_patch(rect)
        # plt.show()
    fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Fin_Img/Image"+str(testIndex)+".csv"
    np.savetxt(fname, tempImg, fmt = '%d', delimiter=',')
    fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Fin_Img2/Image"+str(testIndex)+".csv"
    np.savetxt(fname, tempImg2, fmt = '%d', delimiter=',')
    fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Fin_Img3/Image"+str(testIndex)+".csv"
    np.savetxt(fname, tempImg3, fmt = '%d', delimiter=',')
    testIndex = testIndex + 1