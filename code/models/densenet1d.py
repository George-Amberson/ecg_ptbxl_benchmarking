import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import Flatten
from models.basic_conv1d import create_head1d
###############################################################################################
# Standard resnet

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)



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
            layer = DenseBlock1d(num_features, inplanes, num_layers, kernel_size=kernel_size)
            num_features += num_layers * inplanes
            layers_tmp.append(layer)
            if i != len(layers) - 1:
                layers_tmp.append(nn.BatchNorm1d(num_features))
                layers_tmp.append(nn.ReLU(inplace=True))
                layers_tmp.append(nn.Conv1d(num_features, num_features, kernel_size=1))
                layers_tmp.append(nn.AvgPool1d(kernel_size=2, stride=2))
        
        #head
        layers_tmp.append(nn.BatchNorm1d(num_features))
        layers_tmp.append(nn.ReLU(inplace=True))
        layers_tmp.append(nn.AdaptiveAvgPool1d(1))
        layers_tmp.append(Flatten())
        layers_tmp.append(nn.Linear(num_features, num_classes))

        super(DenseNet1d, self).__init__(*layers_tmp)

    def get_layer_groups(self):
        return (self[6],self[-1])
    
    def get_output_layer(self):
        return self[-1][-1]
        
    def set_output_layer(self,x):
        self[-1][-1]=x

def densenet1d121(**kwargs):
    return DenseNet1d(DenseBlock1d,[2,2,2,2], **kwargs )
