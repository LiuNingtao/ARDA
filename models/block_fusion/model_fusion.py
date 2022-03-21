'''
Author: ningtao liu
Date: 2020-08-29 16:30:56
LastEditors: ningtao liu
LastEditTime: 2020-08-29 20:33:39
FilePath: /ToothAge/block_fusion/model_fusion.py
'''
from numpy.core.shape_base import block
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from efficientnet_pytorch_cus import EfficientNet

class JAWCov(nn.Module):
    def __init__(self, out_channels=3):
        super(JAWCov, self).__init__()
        self.Cov_1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
        self.Cov_2 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 5))
    
    def forward(self, inputs):
        out = F.relu(self.Cov_1(inputs))
        out = F.relu(self.Cov_2(out))
        out = self.max_pool(out)
        return out


class SKULLCov(nn.Module):
    def __init__(self, out_channels=3):
        super(SKULLCov, self).__init__()
        self.Cov_1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
        self.Cov_2 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 3))
    
    def forward(self, inputs):
        out = F.relu(self.Cov_1(inputs))
        out = F.relu(self.Cov_2(out))
        out = self.max_pool(out)
        return out


class BlockFusion(nn.Module):
    def __init__(self, effi_channel=3, num_class=1, is_cuda=True, exchange_factor=0.4):
        super(BlockFusion, self).__init__()

        self.jaw_cov = JAWCov()
        self.skull_cov = SKULLCov()
        
        jaw_model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=effi_channel)
        num_ftrs = 320
        jaw_model._fc = nn.Linear(num_ftrs, num_class)
        self.jaw_model = jaw_model

        skull_model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=effi_channel)
        num_ftrs = 320
        skull_model._fc = nn.Linear(num_ftrs, num_class)
        self.skull_model = skull_model

        self.exchange_factor = exchange_factor

        if is_cuda:
            self.jaw_cov = self.jaw_cov.cuda()
            self.skull_cov = self.skull_cov.cuda()
            self.jaw_model = self.jaw_model.cuda()
            self.skull_model = self.skull_model.cuda()

    def forward(self, inputs):
        jaw_inputs = inputs['jaw']
        skull_inputs = inputs['skull']

        jaw_inputs = self.jaw_cov(jaw_inputs)
        skull_inputs = self.skull_cov(skull_inputs)

        jaw_stem = self.jaw_model._swish(self.jaw_model._bn0(self.jaw_model._conv_stem(jaw_inputs)))
        skull_stem = self.skull_model._swish(self.skull_model._bn0(self.skull_model._conv_stem(skull_inputs)))

        block_length = len(self.jaw_model._blocks)
        for idx in range(block_length):
            drop_connect_rate = self.jaw_model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / block_length
            
            block_jaw = self.jaw_model._blocks[idx]
            block_skull = self.skull_model._blocks[idx]

            jaw_stem = block_jaw(jaw_stem, drop_connect_rate=drop_connect_rate)
            skull_stem = block_skull(skull_stem, drop_connect_rate=drop_connect_rate)
            change_channel_num = int(jaw_stem.size(1) * self.exchange_factor)
            jaw_stem[:, -change_channel_num:, :, :], skull_stem[:, -change_channel_num:, :, :] = skull_stem[:, -change_channel_num:, :, :], jaw_stem[:, -change_channel_num:, :, :]
        
        jaw_output = self.jaw_model._avg_pooling(jaw_stem)
        jaw_output = jaw_output.flatten(start_dim=1)
        jaw_output = self.jaw_model._dropout(jaw_output)
        jaw_output = self.jaw_model._fc(jaw_output)


        skull_output = self.skull_model._avg_pooling(skull_stem)
        skull_output = skull_output.flatten(start_dim=1)
        skull_output = self.skull_model._dropout(skull_output)
        skull_output = self.skull_model._fc(skull_output)

        return jaw_output, skull_output



class HeatFusion(nn.Module):
    def __init__(self, effi_channel=1, num_class=1, is_cuda=True, exchange_factor=0.2):
        super(HeatFusion, self).__init__()

        heat_model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=effi_channel)
        num_ftrs = 320
        heat_model._fc = nn.Linear(num_ftrs, num_class)
        self.heat_model = heat_model

        skull_model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=effi_channel)
        num_ftrs = 320
        skull_model._fc = nn.Linear(num_ftrs, num_class)
        self.skull_model = skull_model

        self.exchange_factor = exchange_factor

        if is_cuda:
            self.skull_model = self.skull_model.cuda()
            self.heat_model = self.heat_model.cuda()

    def forward(self, inputs):
        skull_inputs = inputs['skull']
        heat_inputs = inputs['heat']

        skull_stem = self.skull_model._swish(self.skull_model._bn0(self.skull_model._conv_stem(skull_inputs)))
        heat_stem = self.heat_model._swish(self.heat_model._bn0(self.heat_model._conv_stem(heat_inputs)))

        block_length = len(self.skull_model._blocks)
        for idx in range(block_length):
            drop_connect_rate = self.skull_model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / block_length
            
            block_heat = self.heat_model._blocks[idx]
            block_skull = self.skull_model._blocks[idx]

            heat_stem = block_heat(heat_stem, drop_connect_rate=drop_connect_rate)
            skull_stem = block_skull(skull_stem, drop_connect_rate=drop_connect_rate)
            change_channel_num = int(heat_stem.size(1) * self.exchange_factor)
            heat_stem[:, -change_channel_num:, :, :], skull_stem[:, -change_channel_num:, :, :] = skull_stem[:, -change_channel_num:, :, :], heat_stem[:, -change_channel_num:, :, :]
        
        heat_output = self.heat_model._avg_pooling(heat_stem)
        heat_output = heat_output.flatten(start_dim=1)
        heat_output = self.heat_model._dropout(heat_output)
        heat_output = self.heat_model._fc(heat_output)


        skull_output = self.skull_model._avg_pooling(skull_stem)
        skull_output = skull_output.flatten(start_dim=1)
        skull_output = self.skull_model._dropout(skull_output)
        skull_output = self.skull_model._fc(skull_output)

        return heat_output, skull_output

