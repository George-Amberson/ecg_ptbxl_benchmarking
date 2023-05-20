import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fastai.layers import *
from fastai.core import *
from models.basic_conv1d import create_head1d
###############################################################################################
# Standard resnet

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)
    

class DenseBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=[3,3], bn=True):
        super(DenseBlock1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        if(isinstance(kernel_size,int)): kernel_size = [kernel_size,kernel_size//2+1]
        
        self.layers = nn.ModuleList()
                
        for i in range(num_layers):
            layer = nn.Sequential(
                        nn.BatchNorm1d(in_channels + i*out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(in_channels + i*out_channels, out_channels, kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size[1], padding=(kernel_size[1]-1)//2),
                    )
            self.layers.append(layer)
        
    def forward(self, x):
        features = [x]
        
        for i in range(self.num_layers):
            layer = self.layers[i]
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        return torch.cat(features, dim=1)
    
class DenseBlock1dDO(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=[3,3], bn=True):
        super(DenseBlock1dDO, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        if(isinstance(kernel_size,int)): kernel_size = [kernel_size,kernel_size//2+1]
        
        self.layers = nn.ModuleList()
                
        for i in range(num_layers):
            layer = nn.Sequential(
                        nn.BatchNorm1d(in_channels + i*out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(in_channels + i*out_channels, out_channels, kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size[1], padding=(kernel_size[1]-1)//2),
                        nn.Dropout(p=0.75, inplace=True)
                    )
            self.layers.append(layer)
        
    def forward(self, x):
        features = [x]
        
        for i in range(self.num_layers):
            layer = self.layers[i]
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        
        return torch.cat(features, dim=1)
class DenseNet1d(nn.Sequential):
    def __init__(self, block, layers, kernel_size=3, num_classes=2, input_channels=3, inplanes=64, fix_feature_dim=True, kernel_size_stem=None, stride_stem=2, pooling_stem=True, stride=2, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        self.inplanes = inplanes

        layers_tmp = []

        if(kernel_size_stem is None):
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size,list) else kernel_size
        #stem
        layers_tmp.append(nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem, padding=(kernel_size_stem-1)//2,bias=False))
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))
        if(pooling_stem is True):
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        #backbone
        num_features = inplanes
        for i, num_layers in enumerate(layers):
            layer = block(num_features, inplanes, num_layers, kernel_size=kernel_size)
            num_features += num_layers * inplanes
            layers_tmp.append(layer)
            if i != len(layers) - 1:
                layers_tmp.append(nn.BatchNorm1d(num_features))
                layers_tmp.append(nn.ReLU(inplace=True))
                layers_tmp.append(nn.Conv1d(num_features, num_features, kernel_size=1))
                layers_tmp.append(nn.AvgPool1d(kernel_size=2, stride=2))
        print(num_features)
        #head
        # layers_tmp.append(nn.BatchNorm1d(num_features))
        # layers_tmp.append(nn.ReLU(inplace=True))
        # layers_tmp.append(nn.AdaptiveAvgPool1d(1))
        # layers_tmp.append(Flatten())
        # layers_tmp.append(nn.Linear(num_features, num_classes))
        lin_ftrs=lin_ftrs_head
        nf = num_features
        nc = num_classes
        ps = ps_head
        bn = bn_head
        bn_final =bn_final_head
        lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
        ps = listify(ps)
        if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
        layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
        for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
            layers += bn_drop_lin(ni,no,bn,p,actn)
        print(lin_ftrs[-1])
        if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
        layers_tmp = layers_tmp +layers
        super(DenseNet1d, self).__init__(*layers_tmp)

    def get_layer_groups(self):
        return (self[6],self[-1])
    
    def get_output_layer(self):
        return self[-1][-1]
        
    def set_output_layer(self,x):
        self[-1][-1]=x

def densenet1d2(**kwargs):
    return DenseNet1d(DenseBlock1d,[6,12,24,16], **kwargs )
def densenet1d121(**kwargs):
    return DenseNet1d(DenseBlock1d, [6,12,24,16], **kwargs)
def densenet1d169(**kwargs):
    return DenseNet1d(DenseBlock1d,[6,12,32,32], **kwargs )
def densenet1d201(**kwargs):
    return DenseNet1d(DenseBlock1d,[6,12,48,32], **kwargs )
def densenet1d264(**kwargs):
    return DenseNet1d(DenseBlock1d,[6,12,64,48], **kwargs )

