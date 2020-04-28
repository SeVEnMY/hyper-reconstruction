import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import scipy.ndimage as scin
from scipy import ndimage
from get_name import get_name
import scipy.io as scio
import h5py
new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)

class Hyper_dataset(Dataset):
    def __init__(self, output_shape=512,ratio = 1,Training_mode='Train',data_name = 'CAVE',use_generated_data = False, use_all_data = True):
        self.read_data = use_generated_data
        self.direct_data = use_all_data
        self.data_name = data_name
        assert(output_shape in [32,64,128,256])
        assert(Training_mode in ['Train','Test'])
        assert(self.data_name in ['CAVE','ICVL'])
        self.TM = Training_mode
        self.output_shape = output_shape  
        if self.data_name == 'CAVE':
            self.hsi_data = np.load("/public/aaa/new_hsi_data.npy")
            self.msi_data = np.load('/public/aaa/new_msi_data.npy')
            self.num_pre_img = self.hsi_data.shape[-1]//self.output_shape 
            self.len = self.hsi_data.shape[0] * self.num_pre_img**2 
            self.train_len = round(self.len*0.625)
            self.test_len = round(self.len*0.375)//16
        elif self.data_name == 'ICVL':
            self.hsi_data = np.load("/public/zhanjiang2/data/ICVL/icvl_hsi_5.npy")
            self.msi_data = np.load('/public/zhanjiang2/data/ICVL/icvl_rgb_5.npy')
            self.num_pre_img = self.hsi_data.shape[-1]//self.output_shape 
            self.len = self.hsi_data.shape[0] * self.num_pre_img**2 
            self.train_len = round(self.len*0.7)
            self.test_len = round(self.len*0.3)//16
            self.test_len = self.len-self.train_len
    def __len__(self):
        if self.TM == 'Train':
            return self.train_len
        elif self.TM == 'Test':
            return self.test_len
    def zoom_img(self,input_img,ratio_):
        return np.concatenate([ndimage.zoom(img,zoom = ratio_)[np.newaxis,:,:] for img in input_img],0)
    def recon_img(self, input_img):
        return cv2.resize(cv2.resize(input_img.transpose(1,2,0),dsize=(self.shape1,self.shape1)),dsize = (self.output_shape , self.output_shape)).transpose(2,0,1)
    def __getitem__(self, index):
        if self.TM == 'Test':
            index = index + self.train_len//16
        index_img = index // self.num_pre_img**2 
        index_inside_image = index % self.num_pre_img**2 
        index_row = index_inside_image // self.num_pre_img 
        index_col = index_inside_image % self.num_pre_img
        if self.data_name == 'CAVE':
            if self.TM == 'Train':
                hsi_g = self.hsi_data[index_img,:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]
                msi = self.msi_data[index_img,:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]
                hsi_g = hsi_g.astype(np.float)/(2**16-1)
                msi = msi.astype(np.float)/(2**8-1)
            elif self.TM == 'Test':
                hsi_g = self.hsi_data[index,:,:,:]
                msi = self.msi_data[index,:,:,:]
                hsi_g = hsi_g.astype(np.float)/(2**16-1)
                msi = msi.astype(np.float)/(2**8-1)
        elif self.data_name == 'ICVL':
            if self.TM == 'Train':
                hsi_g = self.hsi_data[index_img,:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]
                msi = self.msi_data[index_img,:,index_row*self.output_shape:(index_row+1)*self.output_shape,index_col*self.output_shape:(index_col+1)*self.output_shape]
                hsi_g = np.reshape(hsi_g,(31,1392,1300))
                msi = np.reshape(msi,(3,1392,1300))
                hsi_g = hsi_g.astype(np.float)/(2**16-1)
                msi = msi.astype(np.float)/(2**8-1)
            elif self.TM == 'Test':
                hsi_g = self.hsi_data[index,:,:,:]
                hsi_g = np.reshape(hsi_g,(31,1392,1300))
                msi = self.msi_data[index,:,:,:]
                si = np.reshape(msi,(3,1392,1300))
                hsi_g = hsi_g.astype(np.float)/(2**16-1)
                msi = msi.astype(np.float)/(2**8-1)
        hsi = hsi_g
        return hsi, msi
        