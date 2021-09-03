import argparse
import time
from os.path import join
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.utils as utils
from data_utils import display_transform

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_path', default=r'D:\UAVLandmark\Dataset\keypoint\BBox_LR', type=str, help='test low resolution image name')
parser.add_argument('--name', default='onechannelx4', type=str, help='where to save the model')
parser.add_argument('--epoch', default=500, type=int, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_PATH = opt.image_path
OUT_PATH = join('results', opt.name,'images_%s'%opt.epoch)
print(OUT_PATH)
os.makedirs(OUT_PATH, exist_ok=True)
MODEL_NAME = join('results', opt.name,'netG_epoch_%d_%d.pth' % (opt.upscale_factor, opt.epoch))


model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))


for name in os.listdir(IMAGE_PATH):
    image = Image.open(os.path.join(IMAGE_PATH, name))
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.clock()
    out = model(image)
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    # import cv2
    # import numpy as np
    # out_img = np.asarray(out_img)
    # out_img = cv2.bilateralFilter(out_img, d=20, sigmaColor=20, sigmaSpace=50)
    # cv2.imwrite(os.path.join(OUT_PATH,  name), out_img)
    # out_img.save(os.path.join(OUT_PATH, 'bilateral', 'out_srf_' + str(UPSCALE_FACTOR) + '_' + name))
    out_img.save(os.path.join(OUT_PATH, name))
    utils.save_image(out, os.path.join(OUT_PATH,'tensor_%s'%(name)))
