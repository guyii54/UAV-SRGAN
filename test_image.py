import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.utils as utils
from data_utils import display_transform

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--image_path', default=r'D:\UAVLandmark\SR\Datasets\LR32_Test_Nopadding', type=str, help='test low resolution image name')
parser.add_argument('--out_path', default=r'D:\UAVLandmark\SR\SRGAN\results\upsample\images',type=str, help='test low resolution image name')
parser.add_argument('--model_name', default=r'D:\UAVLandmark\SR\SRGAN\results\upsample\netG_epoch_8_500.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
IMAGE_PATH = opt.image_path
OUT_PATH = opt.out_path
MODEL_NAME = opt.model_name


model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

import os
os.makedirs(OUT_PATH, exist_ok=True)
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
    import cv2
    import numpy as np
    # out_img = np.asarray(out_img)
    # out_img = cv2.bilateralFilter(out_img, d=20, sigmaColor=20, sigmaSpace=50)
    # cv2.imwrite(os.path.join(OUT_PATH,  name), out_img)
    # out_img.save(os.path.join(OUT_PATH, 'bilateral', 'out_srf_' + str(UPSCALE_FACTOR) + '_' + name))
    out_img.save(os.path.join(OUT_PATH, name))
    utils.save_image(out, os.path.join(OUT_PATH,'tensor_%s'%(name)))
