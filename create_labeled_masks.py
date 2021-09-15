import os
import shutil
import glob
from skimage import io as io
import numpy as np
import cv2
from matplotlib import pyplot as plt

sourcePath = "D:/Ceramica/GUISSONA/Hand Crafted Masks/"

files = os.listdir(sourcePath)
for file in files:
    name = file[:-16]
    print(name)
    
    mask = io.imread(sourcePath+file)[:,:,0]
    flipMask = np.flipud(mask)
    num_labels, labels_im = cv2.connectedComponents(np.uint8(flipMask>200),connectivity=8)
    labels_im = np.flipud(labels_im)
    plt.imsave(sourcePath+name+"_labeledMask.png", labels_im, cmap="gray",vmin=0, vmax=255)