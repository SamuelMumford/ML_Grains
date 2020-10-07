import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from skimage import measure, color, io
import pickle

##Global paramter##
trim_limit = 884
binary_threshold = 100
pixels_to_um = 0.006

##Import tif image##
image_path = r"C:/Users/sammy/Downloads/SEM/SEM/Processed/lrm_grain.tif"
img = cv2.imread(image_path, 0)
plt.imshow(img)
plt.show()

##Basic info##
row_size = np.shape(img)[0]
col_size = np.shape(img)[1]
print("raw size: "+str([row_size, col_size]))

##Labeling clusters on mask##
c4 = [[0,1,0],[1,1,1],[0,1,0]] # 4-connectivity
c8 = [[1,1,1],[1,1,1],[1,1,1]] # 8-connectivity
labeled_mask, num_labels = ndimage.label(img, structure=c4)

print(labeled_mask)

plt.imshow(labeled_mask)
plt.show()

base_img = r"C:/Users/sammy/Downloads/SEM/SEM/Images/img1.tif"
bs = cv2.imread(base_img, 0)
bs = bs[0:img.shape[0], 0:img.shape[1]]
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
l = img.shape[0]//parts
h = img.shape[1]//parts
print(l)
print(h)
trainIndex = 0
testIndex = 0
tol = 1E-5
booly = False
for i in range(0, parts**2):
    for j in range(0, 4):
        b1 = False
        b2 = False
        if(j ==1 or j ==3):
            b1 = True
        if(j==2 or j==3):
            b2 = True
        a = i//parts
        b = i%parts
        ind1 = a*l
        ind2 = (a+1)*l
        ind3 = b*h
        ind4 = (b+1)*h
        imarray = bs[ind1:ind2, ind3:ind4]
        imarray2 = avgd[ind1:ind2, ind3:ind4]
        imarray3 = gradsize[ind1:ind2, ind3:ind4]
        IDarray = labeled_mask[ind1:ind2, ind3:ind4]
        if(b1):
            imarray = imarray[::-1]
            imarray2 = imarray2[::-1]
            imarray3 = imarray3[::-1]
            IDarray = IDarray[::-1]
        if(b2):
            imarray = np.transpose(np.transpose(imarray)[::-1])
            imarray2 = np.transpose(np.transpose(imarray2)[::-1])
            imarray3 = np.transpose(np.transpose(imarray3)[::-1])
            IDarray = np.transpose(np.transpose(IDarray)[::-1])
        tempImg = imarray
        tempImg2 = imarray2
        tempImg3 = imarray3
        objs = np.unique(IDarray)[1:]
        n_obj = len(objs)
        keep = [True]*n_obj
        masks = (IDarray == objs[:, None, None])
        #print('loop iter ' + str(4*i + j))
        #print(n_obj)
        boxes = []
        for q in range(n_obj):
            pos = np.where(masks[q])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if(xmin == xmax or ymin == ymax):
                keep[q] = False
            else:
                boxes.append([xmin, ymin, xmax, ymax])
        objs = objs[keep]
        masks = (IDarray == objs[:, None, None])
        # fig,ax = plt.subplots(1)
        # plt.imshow(masks[-1])
        # # Create a Rectangle patch
        # box = boxes[-1]
        # rect = patches.Rectangle((box[0],box[1]),(box[2]-box[0]),(box[3] - box[1]),linewidth=1,edgecolor='r',facecolor='none')
    
        # # Add the patch to the Axes
        # ax.add_patch(rect)
        # plt.show()
        if(i%partition == 0):
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Test_Img/Image"+str(testIndex)+".csv"
            np.savetxt(fname, tempImg, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Test_Img2/Image"+str(testIndex)+".csv"
            np.savetxt(fname, tempImg2, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Test_Img3/Image"+str(testIndex)+".csv"
            np.savetxt(fname, tempImg3, fmt = '%d', delimiter=',')
            #np.savetxt(fname, imgtog, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Test_Masks/Masks"+str(testIndex)+".csv"
            output = open(fname, 'wb')
            pickle.dump(masks, output)
            output.close()
            #np.savetxt(fname, masks, fmt = '%d', delimiter=',')
            #fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Test_nobjs/nobj"+str(testIndex)+".csv"
            #np.savetxt(fname, n_obj, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Test_boxes/boxes"+str(testIndex)+".csv"
            np.savetxt(fname, boxes, fmt = '%d', delimiter=',')
            testIndex = testIndex + 1
        else:
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Train_Img/Image"+str(testIndex)+".csv"
            np.savetxt(fname, tempImg, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Train_Img2/Image"+str(testIndex)+".csv"
            np.savetxt(fname, tempImg2, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Train_Img3/Image"+str(testIndex)+".csv"
            np.savetxt(fname, tempImg3, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Train_Masks/Masks"+str(testIndex)+".csv"
            #np.savetxt(fname, masks, fmt = '%d', delimiter=',')
            output = open(fname, 'wb')
            pickle.dump(masks, output)
            output.close()
            #fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Train_nobjs/nobj"+str(testIndex)+".csv"
            #np.savetxt(fname, n_obj, fmt = '%d', delimiter=',')
            fname = "C:/Users/sammy/Downloads/SEM/SEM/PFD_Train_boxes/boxes"+str(testIndex)+".csv"
            np.savetxt(fname, boxes, fmt = '%d', delimiter=',')
            trainIndex = trainIndex + 1