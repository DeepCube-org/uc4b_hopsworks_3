# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch import nn


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = torch.nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(4 * dim, dim)
        self.gamma = torch.nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

        if(drop_path>0.):
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__() 
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super(ConvNeXt, self).__init__() 
        assert len(depths)==len(dims), 'The depths and dims parameters must have the same size'

        self.depths = depths
        self.dims = dims


        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        )
        for i in range(len(self.depths)-1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        for i in range(len(self.depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]



        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        if(num_classes is not None):
            self.head = nn.Linear(dims[-1], num_classes)
            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()
            self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            #trunc_normal_(m.weight, std=.02)
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x





def convnext_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnext_small(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnext_base(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnext_large(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnext_xlarge(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model




class PointWiseEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim = 96,
        depths = [2, 2, 4, 2],
        dims = [96//(2**(len([2, 2, 4, 2])-i-1)) for i in range(len([2, 2, 4, 2]))],
        drop_path_rate = 0
     ):
        super(PointWiseEncoder, self).__init__()
        
        assert len(depths)==len(dims), 'Depths and dims must have the same shape.'
        if(len(depths)>0):
            assert dims[-1]==hidden_dim, 'hidden_dim must be equal to dims[-1]'

        input_dim = 3 # (not in order){velocity, mir, coherence}

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.kernel_layer = None
        if(len(depths)>0):
            surf_embedding_dim = 3
            
            in_channels = 2
            in_channels = in_channels+surf_embedding_dim
            in_channels = in_channels+6 # Optical channels used by the model

            self.kernel_layer = torch.nn.Sequential(
                ConvNeXt(num_classes=None, in_chans=in_channels, depths=depths, dims=dims, drop_path_rate = drop_path_rate)
            )
            self.surf_embedding = torch.nn.Embedding(11, surf_embedding_dim, max_norm=1)

            self.merge_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                torch.nn.ReLU(),
            )
        else:
            self.merge_layer = torch.nn.Sequential(torch.nn.ReLU())
        

        

    def patch_forward(self, stgraph):
        if(self.kernel_layer is None):
            return(None)

        attributes_height_kernel = stgraph.attributes_height_kernel.to(self.device)
        attributes_amplitude_kernel = stgraph.attributes_amplitude_kernel.to(self.device)
        attributes_surf_kernel = stgraph.attributes_surf_kernel.to(self.device)

        surf_kernel = self.surf_embedding(attributes_surf_kernel[:, 0, :, :].long()) #(batch, H, W)
        surf_kernel = surf_kernel.permute(0, 3, 1, 2) #(batch, embedding, H, W)
        

        attributes_B4_kernel  = stgraph.attributes_B4_kernel.to(self.device)
        attributes_B3_kernel  = stgraph.attributes_B3_kernel.to(self.device)
        attributes_B2_kernel  = stgraph.attributes_B2_kernel.to(self.device)
        attributes_B8_kernel  = stgraph.attributes_B8_kernel.to(self.device)
        attributes_B11_kernel = stgraph.attributes_B11_kernel.to(self.device)
        attributes_B12_kernel = stgraph.attributes_B12_kernel.to(self.device)

        kernel = torch.cat(
            (
                surf_kernel,
                attributes_height_kernel, 
                attributes_amplitude_kernel,
                attributes_B4_kernel,
                attributes_B3_kernel,
                attributes_B2_kernel,
                attributes_B8_kernel,
                attributes_B11_kernel,
                attributes_B12_kernel
            ), 1
        )

        kernel = kernel.to(self.device)
        kernel = self.kernel_layer(kernel)
        kernel = kernel.reshape(kernel.shape[0], kernel.shape[1])
        return(kernel)


    def pointwise_forward(self, stgraph):

        velocity  = stgraph.attributes_velocity
        coherence = stgraph.attributes_coherence
        mir       = stgraph.attributes_mir

        x = torch.cat((velocity, coherence, mir), -1)
        x = x.to(self.device)
        x = self.input_layer(x)
        return(x)


    def forward(self, stgraph):   
        pointwise_hidden = self.pointwise_forward(stgraph)
        patch_hidden = self.patch_forward(stgraph)
        if(patch_hidden is not None):
            x = torch.cat((pointwise_hidden, patch_hidden), -1)
        else:
            x = pointwise_hidden
        
        x = self.merge_layer(x)
        return(x)

class Data(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Data, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Data, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Data, self).__delitem__(key)
        del self.__dict__[key]



import numpy as np
import os

#tf version: 2.9.1
#torch version: 1.12.0+cu102
class Predict:

    def __init__(self):
        """ Initialization code goes here:
            - Download the model artifact
            - Load the model
        """
        self.model = PointWiseEncoder()
        self.model.load_state_dict(torch.load(os.environ["ARTIFACT_FILES_PATH"] + '/model.pkl'), strict=False)
        self.model.device = 'cpu'

    def predict(self, inputs):
        """ Serve predictions using the trained model"""
        
        outputs = []

        for input in inputs:
            for key in input.keys():
                input[key] = torch.from_numpy(np.array(input[key]).astype(np.float32))
            input = Data(**input)
            output = self.model(input).detach().numpy().tolist()
            outputs.append(output)
        
        return(outputs)


#attributes_height_kernel, attributes_amplitude_kernel, attributes_surf_kernel
#attributes_B4_kernel, attributes_B3_kernel, attributes_B2_kernel, attributes_B8_kernel, attributes_B11_kernel, attributes_B12_kernel
#attributes_velocity, attributes_coherence, attributes_mir




if __name__ == '__main__':
    
    model = PointWiseEncoder()
    model.device = 'cpu'
    
    size = 80

    input = {
        'attributes_height_kernel'    : torch.ones((100, 1, size, size)),
        'attributes_amplitude_kernel' : torch.ones((100, 1, size, size)),
        'attributes_surf_kernel'      : torch.ones((100, 1, size, size)),

        'attributes_B4_kernel': torch.ones((100, 1, size, size)),
        'attributes_B3_kernel': torch.ones((100, 1, size, size)),
        'attributes_B2_kernel': torch.ones((100, 1, size, size)),
        'attributes_B8_kernel': torch.ones((100, 1, size, size)),
        'attributes_B11_kernel': torch.ones((100, 1, size, size)),
        'attributes_B12_kernel': torch.ones((100, 1, size, size)),

        'attributes_velocity': torch.ones((100, 1)),
        'attributes_coherence': torch.ones((100, 1)),
        'attributes_mir': torch.ones((100, 1))
    }
    data = Data(**input)

    output = model(data)
    print(output.shape)
    output = output.detach().numpy().tolist()
    
    exit(0)
    
    print(len(output))