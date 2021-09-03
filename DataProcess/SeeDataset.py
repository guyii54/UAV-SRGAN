import os
from os.path import join
import cv2
import numpy as np

def seeAnno():
    dataroot = r'D:\UAVLandmark\Dataset\data903\rawdata'
    annoroot = r'D:\UAVLandmark\Dataset\data903\UAV_keypoint_train.npy'
    anno = np.load(annoroot, allow_pickle=True)
    print(anno)


if __name__ == '__main__':
    seeAnno()

