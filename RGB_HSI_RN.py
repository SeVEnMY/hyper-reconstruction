import torch.nn as nn
from Hyper_loader import Hyper_dataset
import SobelConv
from torch.utils import data
import argparse
import torch
import cv2
import ResNet
from torch.autograd import Variable 
from torchnet.logger import VisdomPlotLogger, VisdomLogger,VisdomTextLogger
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as x_init
import matplotlib
from tensorboardX import SummaryWriter
import skimage.measure as skm
import pytorch_ssim
import SpectralResponseFunction

from scipy import ndimage as ndi

import math

from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

parse = argparse.ArgumentParser()
parse.add_argument('--n_epochs', type = int, default = 5000, help = 'number of epochs of training')
parse.add_argument('--batch_size',type = int, default = 8, help = 'size of the batches')
parse.add_argument('--lr',type = float, default = 1e-3, help='learing rate of network')
parse.add_argument('--b1',type = float, default = 0.9, help='adam:decay of first order momentum of gradient')
parse.add_argument('--b2',type = float, default = 0.999, help= 'adam: decay of first order momentum of gradient')
opt = parse.parse_args()

MseLoss = torch.nn.MSELoss()

def psnr(img1, img2):
    mse = MseLoss(img1, img2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay=[0,0], p =2,batch_size = 8):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.bz  = batch_size
    def to(self,device):
        self.device = device
        super().to(device)
        return self
    def forward(self,model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay)
        return reg_loss
    
    def get_weight(self,model):

        weight_list = []

        for name,param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay):
        reg_loss = 0
        for name,w in weight_list:
            length = list(w.shape)[-3]
            drivate_matrix = torch.eye(length)
            temp = torch.zeros_like(drivate_matrix)
            temp[1:,:] = drivate_matrix[:-1,:]
            drivate_matrix  = drivate_matrix - temp
            drivate_matrix[-1,-1] = 0
            drivate_matrix = drivate_matrix.cuda()
            w = w.reshape(-1,w.shape[-3])
            reg_loss = reg_loss + (torch.sum((w.mm(drivate_matrix))**2) / (list(w.reshape(-1).shape)[0] - list(w.shape)[0])).sqrt()*weight_decay[0]
            reg_loss = reg_loss + (torch.sum(w**2) / list(w.reshape(-1).shape)[0]).sqrt()  *weight_decay[1]

            return reg_loss

class RevertNet(nn.Module):

    def __init__(self):
        super(RevertNet, self).__init__()
        self.conv1 = nn.Conv2d(31, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out

Hydata = Hyper_dataset(output_shape=128, Training_mode='Train')
Valdata = Hyper_dataset(output_shape=128, Training_mode='Test')

batch_size = opt.batch_size
train_loader = torch.utils.data.DataLoader(Hydata, batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(Valdata, batch_size=1)

hypernet = ResNet.resnet32().cuda().float()
sobel = SobelConv.SobelConv().cuda().float()

downsample = SpectralResponseFunction.SpRes().cuda().float()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

reg_loss = Regularization(model = hypernet, weight_decay=[0.001,0.001])
ssim = pytorch_ssim.SSIM()

if __name__ == '__main__':
    optimizer = torch.optim.Adam(hypernet.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2),weight_decay=0.001)

    b_r=0
    loss_list = []
    val_ssim_list = []
    val_psnr_list =[]
    val_iters = 0
    val_iters_list = []
    iters = 0
    iter_list = []
    best_ssim = -100000
    best_psnr = -100000
    best_mse = 100000

    writer = SummaryWriter()

    for epoch in range(1, opt.n_epochs+1):
        batch = 0
        loss_ = []
        for hsi, msi in train_loader:
            iters = iters+1
            batch = batch+1

            hsi = Variable(hsi.cuda().float(),requires_grad = False) 
            msi = Variable(msi.cuda().float(),requires_grad = False)

            hypernet.train()

            optimizer.zero_grad()
                
            x = hypernet(msi)
            output = x.cpu().detach().numpy().astype(np.uint16)
            gt = hsi.cpu().detach().numpy().astype(np.uint16)

            ssim_loss = ssim(x, hsi)
            psnr_loss = psnr(x, hsi)
            total_loss = ssim_loss*100 + psnr_loss*1.5

            downsampling = downsample(x)
            downsample_loss = MseLoss(downsampling, msi)

            if epoch > 2999:
                edges1 = sobel(x)
                edges2 = sobel(hsi)
                edgeLoss = MseLoss(edges1, edges2)

                loss = - total_loss + 0.5*reg_loss(hypernet) + downsample_loss + edgeLoss
            else:
                loss = - total_loss + 0.5*reg_loss(hypernet) + downsample_loss

            loss.backward()
            
            optimizer.step()

            for p__ in hypernet.parameters():
            	p__.data.clamp_(0)

            loss_.append(loss.item())
            print("[Epoch %d/5000][Batch:%d/39][loss:%f][SSIM:%f][psnr:%f]"%(epoch,batch,loss.item(),ssim_loss,psnr_loss))
            writer.add_scalar('Loss/train', loss, iters)
            loss_list.append(np.mean(np.array(loss_)))
            iter_list.append(iters)

        if epoch%20 == 0:
            hypernet.eval()
            with torch.no_grad():
                for hsi, msi in validation_loader:
                    hsi = Variable(hsi.cuda().float(), requires_grad = False) 
                    msi = Variable(msi.cuda().float(), requires_grad = False)
                    val_iters = val_iters+1
                    x = hypernet(msi)
                    output = x.cpu().detach().numpy().astype(np.uint16)
                    gt = hsi.cpu().detach().numpy().astype(np.uint16)

                    val_ssim = ssim(x, hsi)
                    val_psnr = psnr(x, hsi)

                    writer.add_scalar('SSIM', val_ssim, val_iters)
                    writer.add_scalar('PSNR', val_psnr, val_iters)

                    print("Validation: %d [SSIM:%f][PSNR:%f]"%(val_iters,val_ssim, val_psnr))
                    if val_ssim > best_ssim:
                        best_ssim = val_ssim
                        torch.save(hypernet.state_dict(), '/public/zhanjiang2/models/torch-hyper-apr-20/best_checkpoint.pkl')
                    if val_psnr > best_psnr:
                        best_psnr = val_psnr

                    print("The highest ssim is: %f"%(best_ssim))
                    print("The highest psnr is: %f"%(best_psnr))

                    writer.add_scalar('Best SSIM', best_ssim, val_iters)
                    writer.add_scalar('Best PSNR', best_psnr, val_iters)
    writer.close()
