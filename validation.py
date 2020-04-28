import torch.nn as nn
from Hyper_loader import Hyper_dataset
import SobelConv
from torch.utils import data
import argparse
import torch
import cv2
import ResNet
import nonLocalConcatenation
from torch.autograd import Variable 
from torchnet.logger import VisdomPlotLogger, VisdomLogger,VisdomTextLogger
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as x_init
import matplotlib
from tensorboardX import SummaryWriter
import pytorch_ssim

from scipy import ndimage as ndi

from PIL import Image


Valdata = Hyper_dataset(output_shape=128, ratio=1/8, use_generated_data=True, Training_mode='Test')
validation_loader = torch.utils.data.DataLoader(Valdata, batch_size=16)

hypernet = ResNet.resnet32().cuda().float()
checkpoint = torch.load('/public/zhanjiang2/models/torch-hyper/cifar10/WRN-2/model_best.pth.tar')
wrn.load_state_dict(checkpoint['state_dict'])

hypernet.eval()
with torch.no_grad():
    for hsi, msi in validation_loader:
        hsi = Variable(hsi.cuda().float(),requires_grad = False) 
        msi = Variable(msi.cuda().float(),requires_grad = False)
        x = hypernet(msi)
        val_loss = ssim_loss(x, hsi)
        print("The loss is: %f"%(val_loss))