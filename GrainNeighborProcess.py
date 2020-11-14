# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:38:14 2020

@author: sammy
"""

import numpy as np
from scipy import ndimage
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from scipy import ndimage
import copy


def expdec(x, a, b):
    return a*np.exp(-x/b)

fname = "C:/Users/sammy/Downloads/SEM/SEM/PredictMask4_img2.csv"
preds = np.loadtxt(fname, dtype = int, delimiter=',')
parts = 7

def compareLists(base, l1, l2):
    return [(i!=j) and (i!=k) for i, j, k in zip(base, l1, l2)]

def row_process(mat, parts):
    l = mat.shape[0]/parts
    l = int(l)
    for i in range(1, parts):
        rowBase = mat[i*l]
        rowT1 = mat[i*l-2]
        rowT2 = mat[i*l-3]
        flip = compareLists(rowBase, rowT1, rowT2)
        mat[i*l, flip] = 1 - mat[i*l, flip]
        
        rowBase = mat[i*l]
        rowT1 = mat[i*l+1]
        rowT2 = mat[i*l+2]
        flip = compareLists(rowBase, rowT1, rowT2)
        mat[i*l, flip] = 1 - mat[i*l, flip]
        
        rowBase = mat[i*l]
        rowT1 = mat[i*l-2]
        rowT2 = mat[i*l+1]
        flip = compareLists(rowBase, rowT1, rowT2)
        mat[i*l, flip] = 1 - mat[i*l, flip]

        
        rowBase = mat[i*l-1]
        rowT1 = mat[i*l+1]
        rowT2 = mat[i*l+2]
        flip = compareLists(rowBase, rowT1, rowT2)
        mat[i*l-1, flip] = 1 - mat[i*l-1, flip]
        
        rowBase = mat[i*l-1]
        rowT1 = mat[i*l-2]
        rowT2 = mat[i*l-3]
        flip = compareLists(rowBase, rowT1, rowT2)
        mat[i*l-1, flip] = 1 - mat[i*l-1, flip]
        
        rowBase = mat[i*l-1]
        rowT1 = mat[i*l-2]
        rowT2 = mat[i*l]
        flip = compareLists(rowBase, rowT1, rowT2)
        mat[i*l-1, flip] = 1 - mat[i*l-1, flip]
    return mat

pfilt = np.copy(preds)
pfilt = row_process(pfilt, parts)
pfilt = row_process(np.transpose(pfilt), parts)
pfilt = np.transpose(pfilt)


plt.imshow(pfilt)
plt.savefig('ModelMaskImg1.png')
plt.show()

##Labeling clusters on mask##
c4 = [[0,1,0],[1,1,1],[0,1,0]] # 4-connectivity
c8 = [[1,1,1],[1,1,1],[1,1,1]] # 8-connectivity
labeled_pred, num_labels = ndimage.label(pfilt, structure=c4)
maxi = np.max(labeled_pred)
print(maxi)

plt.imshow(labeled_pred)
plt.show()

def getAreas(labeled_mask):
    objs = np.unique(labeled_mask)[1:]
    n_obj = len(objs)
    areas = [0]*n_obj
    pixAreas = [0]*n_obj
    masks = (labeled_mask == objs[:, None, None])
    for q in range(n_obj):
        pos = np.where(masks[q])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        pixAreas[q] = 1.0*len(pos[0])
        area = 1.0*(xmax - xmin)*(ymax - ymin)
        areas[q] = max(area, 1.0)
    return areas, np.sqrt(pixAreas)*np.sqrt(4/np.pi), pixAreas

predAreas, predLs, predPixA = getAreas(labeled_pred)

predLs = predLs*.006

struct2 = ndimage.generate_binary_structure(2, 2)
pfiltB = ndimage.binary_dilation(pfilt, structure=struct2).astype(pfilt.dtype)
pfiltB -= pfilt
plt.imshow(pfiltB[0:200, 0:200])
plt.savefig('ModelMaskBorder.png')
plt.show()

grain_dict = {new_dict: {} for new_dict in range(1,maxi+1)}

lp = copy.deepcopy(labeled_pred)
notseen = (lp >= 0)
j =0
look = True
while look:
    for i in range(1, maxi+1):
        #print(i)
        background = (lp == 0)
        obj = (lp == i)
        border = np.logical_and(ndimage.binary_dilation(obj, structure=struct2), np.invert(obj))
        lp += background*border*i
        mask = np.logical_and(np.logical_and(border, np.invert(background)), notseen)
        sub = np.ma.MaskedArray(lp, np.invert(mask)).compressed()
        neigh, counts = np.unique(sub, return_counts=True)
        for p in range(0, len(neigh)):
            bl = counts[p]
            booly = (neigh[p] in grain_dict[i].keys())
            if(not booly):
                if(i < neigh[p]):
                    dist = 2*j + 1
                else:
                    dist = 2*j + 2
                grain_dict[i][neigh[p]] = [dist, bl]
                grain_dict[neigh[p]][i] = [dist, bl]
            else:
                grain_dict[i][neigh[p]][1] += bl
                grain_dict[neigh[p]][i][1] += bl
        #print(sub)
    j += 1
    print(j)
    background = (lp == 0)
    crit = np.ma.MaskedArray(lp, np.invert(background)).compressed().shape[0]
    print(crit)
    if(crit == 0):
        look = False
plt.imshow(lp)
plt.show()
print(grain_dict)    