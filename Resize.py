import cv2
from PIL import Image as I
import numpy as np

img_file = r'D:\UAVLandmark\fig\backlight_backlight40%_overexp_0161.jpg'
img = cv2.imread(img_file)
img_resize = cv2.resize(img, (0,0), fx=1/8, fy=1/8)
cv2.imshow('a',img_resize)
cv2.waitKey()
a = cv2.imwrite(r'D:\UAVLandmark\fig\backlight_backlight40%_overexp_0161_res.jpg',img_resize)