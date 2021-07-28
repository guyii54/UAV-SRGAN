import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.utils as utils
from data_utils import display_transform

from model import Generator

parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--image_path', default=r'D:\UAVLandmark\SR\Deep-Iterative-Collaboration\code\Datasets\LR32_Test_Nopadding', type=str, help='test low resolution image name')
parser.add_argument('--out_path', default=r'D:\UAVLandmark\SR\SRGAN\outputs\32x32Nopadding',type=str, help='test low resolution image name')
opt = parser.parse_args()

UPSCALE_FACTOR = 8  # 超分尺度
IMAGE_PATH = opt.image_path
OUT_PATH = opt.out_path

model = Generator(UPSCALE_FACTOR).eval()
model.cuda()
model.load_state_dict(torch.load('netG_epoch_8_500.pth'))   #加载模型


import os
os.makedirs(OUT_PATH, exist_ok=True)
for name in os.listdir(IMAGE_PATH):
    image = Image.open(os.path.join(IMAGE_PATH, name))
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    image = image.cuda()

    out = model(image)

    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(os.path.join(OUT_PATH, name))
    utils.save_image(out, os.path.join(OUT_PATH,'tensor_%s'%(name)))
