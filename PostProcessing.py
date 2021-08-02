import cv2
from os.path import join
import os

exper_name = r'upsample'
post_name = 'bilateralFilter'

in_dir = join('results',exper_name,'images_500')
out_dir = join('results','post',post_name+'_'+exper_name)
os.makedirs(out_dir, exist_ok=True)
img_list = os.listdir(in_dir)
for name in img_list:
    img = cv2.imread(join(in_dir, name))
    outimg = cv2.bilateralFilter(img, d=20, sigmaColor=20, sigmaSpace=50)
    cv2.imwrite(join(out_dir,name), outimg)