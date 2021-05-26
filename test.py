import os
import sys
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from os.path import join, isdir, abspath, dirname

# Customized import.
from networks import HED
from datasets import BsdsDataset
from utils import Logger, AverageMeter, \
    load_checkpoint, save_checkpoint, load_vgg16_caffe, load_pretrained_caffe

current_dir = abspath(dirname(__file__))
output_dir = join(current_dir, './test_output_1')
device = torch.device('cuda')
net = nn.DataParallel(HED(device))
net.to(device)

net_parameters_id = defaultdict(list)
for name, param in net.named_parameters():
        if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                    'module.conv2_1.weight', 'module.conv2_2.weight',
                    'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                    'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
            print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id['conv1-4.weight'].append(param)
        elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                      'module.conv2_1.bias', 'module.conv2_2.bias',
                      'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                      'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
            print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id['conv1-4.bias'].append(param)
        elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
            print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id['conv5.weight'].append(param)
        elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias'] :
            print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id['conv5.bias'].append(param)
        elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                      'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
            print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id['score_dsn_1-5.weight'].append(param)
        elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                      'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
            print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id['score_dsn_1-5.bias'].append(param)
        elif name in ['module.score_final.weight']:
            print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id['score_final.weight'].append(param)
        elif name in ['module.score_final.bias']:
            print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id['score_final.bias'].append(param)   

lr = 1e-6

opt = torch.optim.SGD([
        {'params': net_parameters_id['conv1-4.weight']      , 'lr': lr*1    , 'weight_decay': 2e-4},
        {'params': net_parameters_id['conv1-4.bias']        , 'lr': lr*2    , 'weight_decay': 0.},
        {'params': net_parameters_id['conv5.weight']        , 'lr': lr*100  , 'weight_decay': 2e-4},
        {'params': net_parameters_id['conv5.bias']          , 'lr': lr*200  , 'weight_decay': 0.},
        {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': lr*0.01 , 'weight_decay': 2e-4},
        {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': lr*0.02 , 'weight_decay': 0.},
        {'params': net_parameters_id['score_final.weight']  , 'lr': lr*0.001, 'weight_decay': 2e-4},
        {'params': net_parameters_id['score_final.bias']    , 'lr': lr*0.002, 'weight_decay': 0.},
    ], lr=lr, momentum=0.9, weight_decay=2e-4)

load_checkpoint(net,opt, 'E:/2021-04/HED/hed_pytorch/output/epoch-36-checkpoint.pt')

test_dataset  = BsdsDataset(dataset_dir='./data/TDP', split='test_1')
test_loader   = DataLoader(test_dataset,  batch_size=1,
                               num_workers=0, drop_last=False, shuffle=False)

def test(test_loader, net, save_dir):
    """ Test procedure. """
    # Create the directories.
    if not isdir(save_dir):
        os.makedirs(save_dir)
    save_png_dir = join(save_dir, 'png')
    if not isdir(save_png_dir):
        os.makedirs(save_png_dir)
    save_mat_dir = join(save_dir, 'mat')
    if not isdir(save_mat_dir):
        os.makedirs(save_mat_dir)
    # Switch to evaluation mode.
    net.eval()
    # Generate predictions and save.
    # assert args.test_batch_size == 1  # Currently only support test batch size 1.
    for batch_index, images in enumerate(tqdm(test_loader)):
        images = images.cuda()
        _, _, h, w = images.shape
        preds_list = net(images)
        fuse       = preds_list[-1].detach().cpu().numpy()[0, 0]  # Shape: [h, w].
        name       = test_loader.dataset.images_name[batch_index]
        sio.savemat(join(save_mat_dir, '{}.mat'.format(name)), {'result': fuse})
        Image.fromarray((fuse * 255).astype(np.uint8)).save(join(save_png_dir, '{}.png'.format(name)))
        # print('Test batch {}/{}.'.format(batch_index + 1, len(test_loader)))


if __name__ == "__main__":
    test(test_loader, net, save_dir=join(output_dir, 'test'))