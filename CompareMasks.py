# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:38:14 2020

@author: sammy
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit

def expdec(x, a, b):
    return a*np.exp(-x/b)

fname = "C:/Users/sammy/Downloads/SEM/SEM/PredictMaskSher3dpt2_imgQiZoom.csv"
preds = np.loadtxt(fname, dtype = int, delimiter=',')
fname = "C:/Users/sammy/Downloads/SEM/SEM/PredictMask4.csv"
real = np.loadtxt(fname, dtype = int, delimiter=',')
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
print(np.mean((real-pfilt)**2))

plt.imshow(pfilt)
plt.savefig('ModelMask2.png')
plt.show()
plt.imshow(real)
plt.savefig('TrueMask2.png')
plt.show()
plt.imshow((real-pfilt)**2)
plt.savefig('Error2.png')
plt.show()
zer = np.zeros(real.shape)
combo = np.stack((real*255, 255*(real-pfilt)**2, zer), axis = 2)
comboImg = Image.fromarray(combo.astype(np.uint8))
comboImg.show()

##Labeling clusters on mask##
c4 = [[0,1,0],[1,1,1],[0,1,0]] # 4-connectivity
c8 = [[1,1,1],[1,1,1],[1,1,1]] # 8-connectivity
labeled_pred, num_labels = ndimage.label(pfilt, structure=c4)
labeled_mask, num_labels = ndimage.label(real, structure=c4)

print(np.max(labeled_pred))
print(np.max(labeled_mask))

plt.imshow(labeled_pred)
plt.show()
plt.imshow(labeled_mask)
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
        pixAreas[q] = len(pos[0])
        area = 1.0*(xmax - xmin)*(ymax - ymin)
        areas[q] = max(area, 1.0)
    return areas, np.sqrt(pixAreas), pixAreas

predAreas, predLs, predPixA = getAreas(labeled_pred)
realAreas, realLs, realPixA = getAreas(labeled_mask)

#print(predPixA)
#print(realPixA)

mini = min(np.min(realLs), np.min(predLs))
maxi = max(np.max(realLs), np.max(predLs))
nbins = 25
bins = np.linspace(mini, maxi, nbins)

midBins = (bins[1:] + bins[:-1])/2
predCount = np.histogram(predLs, bins=bins)[0]
trueCount = np.histogram(realLs, bins=bins)[0]
# valsPred, covarPred = curve_fit(expdec, midBins, predCount, p0=[max(predCount), 5], sigma = (np.sqrt(predCount) + 1))
# valsTrue, covarTrue = curve_fit(expdec, midBins, trueCount, p0=[max(predCount), 5], sigma = (np.sqrt(predCount) + 1))

# print('Predicted and real fit coef.')
# print(valsPred)
# print(valsTrue)
# sigPred = np.sqrt(np.diag(covarPred))
# sigTrue = np.sqrt(np.diag(covarTrue))
# print('Predicted and real fit std. dev.')
# print(sigPred)
# print(sigTrue)
# print('Difference in fit coef. of:')
# print((valsPred - valsTrue)/(np.sqrt(sigPred**2 + sigTrue**2)))


plt.hist(predLs, bins, alpha=0.5, label='img1')
#plt.hist(realLs, bins, alpha=0.5, label='img2')
#plt.plot(midBins, expdec(midBins, valsPred[0], valsPred[1]), label = "pred fit")
#plt.plot(midBins, expdec(midBins, valsTrue[0], valsTrue[1]), label = "real fit")
plt.legend(loc='upper right')
plt.xlabel('grain l scale')
plt.ylabel('counts')
plt.show()

plt.hist(predLs, bins, alpha=0.5, label='img1')
#plt.hist(realLs, bins, alpha=0.5, label='img2')
#plt.plot(midBins, expdec(midBins, valsPred[0], valsPred[1]), label = "pred fit")
#plt.plot(midBins, expdec(midBins, valsTrue[0], valsTrue[1]), label = "real fit")
plt.legend(loc='upper right')
plt.xlabel('grain l scale')
plt.ylabel('counts')
#plt.ylim(ymin=.8)
plt.yscale('log')
plt.show()

nbins = 30
binsl = np.logspace(np.log10(mini), np.log10(maxi), nbins)

plt.hist(predLs, binsl, alpha=0.5, label='img1')
#plt.hist(realLs, binsl, alpha=0.5, label='img2')
plt.legend(loc='upper right')
plt.xlabel('log10 grain lengthscale')
plt.ylabel('counts')
#plt.yscale('log')
plt.xscale('log')
plt.show()

nbins = 15
binsl = np.logspace(np.log10(mini), np.log10(maxi), nbins)

plt.hist(predLs, binsl, alpha=0.5, label='img1')
#plt.hist(realLs, binsl, alpha=0.5, label='img2')
plt.legend(loc='upper right')
plt.xlabel('log10 grain lengthscale')
plt.ylabel('counts')
#plt.yscale('log')
plt.xscale('log')
plt.show()