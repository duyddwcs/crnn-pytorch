# -*- coding: utf-8 -*-
import os.path
import random
import torch
import numpy as np

HOME = os.path.expanduser("~")

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
## case sensitive
# alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

keep_ratio = False   # whether to keep ratio for image resize
random_sample = True # whether to sample the dataset with random sampler
imgH = 32            # the height of the input image 
imgW = 100           # the width of the input image 
nh = 256             # size of the lstm hidden state
nc = 1               # number of channels

pretrained = ''     # path to pretrained model (to continue training)
expr_dir = 'expr'   # where to store samples and models
trainRoot = './synth90k/train'        # path to the dataset
valRoot = './synth90k/train'          # path to the dataset
testRoot = ''                         # path to the dataset
model_path = ''                       # path to trained model

# hardware
cuda = True             # enables cuda
multi_gpu = False       # whether to use multi gpu
ngpu = 1                # number of GPUs to use when use multi gpu
workers = 0             # number of data loading workers

# training process
displayInterval = 100   # interval to be print the train loss
valInterval = 1000      # interval to val the model loss and accuray
saveInterval = 1000     # interval to save model
n_val_disp = 10         # number of samples to display when validate the model

nepoch = 1000          
batchSize = 64         
lr = 0.0001           
beta1 = 0.5             # beta1 for adam
adam = False            # whether to use adam (default is rmsprop)
adadelta = False        # whether to use adadelta (default is rmsprop)

