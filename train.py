import argparse
import os
from os.path import join
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from loss import TVLoss
from loss import PercepLoss
from torch.nn import MSELoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=8, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=500, type=int, help='train epoch number')
parser.add_argument('--name', type=str, help='where to save the model')
parser.add_argument('--data_dir', default='None', type=str, help='where to save the model')



if __name__ == '__main__':
    data_root = r'D:\UAVLandmark\Dataset\keypoint\data'
    train_annos = r'D:\UAVLandmark\Dataset\keypoint\annotations\UAV_train.npy'
    val_annos = r'D:\UAVLandmark\Dataset\keypoint\annotations\UAV_test.npy'
    # train_dir = r'D:\UAVLandmark\Dataset\data624\HR3'
    # val_dir = r'D:\UAVLandmark\SR\Datasets\HR_Test'
    losses = ['MSE', 'g', 'd', 'TV']

    opt = parser.parse_args()
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    experiment_name = opt.name

    save_dir = os.path.join('results', experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_set = TrainDatasetFromFolder(data_root, train_annos, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(data_root, val_annos, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(UPSCALE_FACTOR, inchannel=1)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator(inchannel=1)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss(inchannel=1)
    MSE_criterion = MSELoss()
    TV_criterion = TVLoss()
    Keypoint_criterion = PercepLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    # netG.load_state_dict(torch.load('epochs/netG_epoch_8_500.pth'))
    results = {}
    for l in losses:
        results['%s_loss'%l] = []
    results['d_score'] = []
    results['g_score'] = []
    results['psnr'] = []
    results['ssim'] = []
    # results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        # running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        running_results = {}
        for l in losses:
            key_name = '%s_loss'%l
            running_results[key_name] = 0
        running_results['batch_sizes'] = 0
        running_results['d_score'] = 0
        running_results['g_score'] = 0

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)
    
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img)
            MSE_loss = MSE_criterion(fake_img, real_img)
            TV_loss = TV_criterion(fake_img)
            Keypoint_loss = Keypoint_criterion(fake_img, real_img)
            total_loss = g_loss
            # print(MSE_loss, g_loss, TV_loss)
            total_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['MSE_loss'] += MSE_loss.item() * batch_size
            running_results['TV_loss'] += TV_loss.item() * batch_size

            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_MSE: %.4f Loss_TV: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS,
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['MSE_loss'] / running_results['batch_sizes'],
                running_results['TV_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            # count = 0
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
                # utils.save_image(sr, 'outputs/test_during_train/SR%d.png'%(count))
                # utils.save_image(lr, 'outputs/test_during_train/LR%d.png'%(count))
                # count+=1

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            # training_save = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
            training_save = join(save_dir,'training_save')
            if not os.path.exists(training_save):
                os.makedirs(training_save)
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image,  join(training_save,'epoch_%d_index_%d.png' % (epoch, index)), padding=5)
                index += 1
    
        # save model parameters
        if epoch%1 == 0:
            torch.save(netG.state_dict(), join(save_dir, 'netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
            torch.save(netD.state_dict(), join(save_dir, 'netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['MSE_loss'].append(running_results['MSE_loss'] / running_results['batch_sizes'])
        results['TV_loss'].append(running_results['TV_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            # out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'],
                      'Loss_G': results['g_loss'],
                      'Loss_MSE': results['MSE_loss'],
                      'Loss_TV': results['TV_loss'],
                      'Score_D': results['d_score'],
                      'Score_G': results['g_score'],
                      'PSNR': results['psnr'],
                      'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(save_dir + '/srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
