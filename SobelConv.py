import torch
from torch import nn

class SobelConv(nn.Module):
    def __init__(self, in_channel = 31, batch_num = 16):
        super(SobelConv,self).__init__()
        self.bz = batch_num
        self.in_channel = in_channel
        self.convx = nn.Conv2d(in_channels = 31 , out_channels  =31, kernel_size = 3, stride= 1, bias= False, padding= 1)
        self.convy = nn.Conv2d(in_channels = 31,  out_channels = 31, kernel_size = 3, stride= 1, bias= False, padding= 1)
    def init_weight(self):
        temp_x = torch.tensor([[-1,0,-1],[2,0,2],[-1,0,-1]])
        temp_y = torch.tensor([[-1,2,-1],[0,0,0],[-1,2,-1]])
        param_x = torch.zeros_like(self.convx.weight)
        param_y = torch.zeros_like(self.convy.weight)
        for index in range(self.in_channel):
            param_x[index, index, :,:] = temp_x
            param_y[index,index, :, :] =temp_y
        self.convx.weight = nn.Parameter(param_x)
        self.convy.weight = nn.Parameter(param_y)

    def forward(self,input_x):
        edge_x = self.convx(input_x)
        edge_y = self.convy(input_x)
        edge = edge_x.abs()+edge_y.abs()
        return edge
