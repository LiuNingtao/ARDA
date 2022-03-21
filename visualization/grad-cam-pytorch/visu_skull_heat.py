#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from torch.autograd import Variable
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import pandas as pd
import time
import sys
import gc
import time
import json

from torchvision.transforms import functional as F

from grad_cam_skull import BackPropagation, Deconvnet, GradCAM, GuidedBackPropagation
from efficientnet_pytorch import EfficientNet

from customer_trans import FixShape, ToImage, Enhancement, ToTensor
import heapq

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

skull_heat_map_file = open(r'')
skull_heat_map = json.load(skull_heat_map_file)

def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(heat_path, map_path, npy_path, gcam, raw_image):
    raw_image = np.expand_dims(raw_image, axis=2)
    raw_image = np.concatenate((raw_image, raw_image, raw_image), axis=-1)
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    np.save(npy_path, gcam)
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    cv2.imwrite(heat_path, np.uint8(gcam))

    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(map_path, np.uint8(gcam))
    del gcam


def load_data(path):
    image = sitk.ReadImage(path)
    image_data = sitk.GetArrayFromImage(image)
    image_data = np.squeeze(image_data)
    # if len(image_data.shape) != 3:
    #     image_data = np.squeeze(image_data)
    #     image_data = image_data[np.newaxis, :, :]
    if len(image_data.shape) > 2:
        shape_list = image_data.shape
        # 最大的2个数对应的，如果用nsmallest则是求最小的数及其索引
        max_index = map(shape_list.index, heapq.nlargest(2, shape_list))
        # max_index 直接输出来不是数，使用list()或者set()均可输出
        max_index = set(max_index)
        if max_index == {0, 1}:
            image_data = image_data[..., 0]
        elif max_index == {1, 2}:
            image_data = image_data[0, ...]
    return image_data



def enhancement(image_data: np.ndarray):
    image_data.astype(np.int16)
    image_normalized = np.zeros(image_data.shape, np.uint8)
    cv2.normalize(image_data, image_normalized, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    clahe =cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    dst = clahe.apply(image_normalized)
    return dst

def fix_shape(item: np.ndarray):
    assert len(item.shape) == 2, print(item.shape)
    if item.shape[0] < item.shape[1]:
        pad_num = item.shape[1] - item.shape[0]
        if pad_num % 2 == 0:
            padded_array = np.pad(item, pad_width=((pad_num // 2, pad_num // 2), (0, 0)))
        else:
            padded_array = np.pad(item, pad_width=((pad_num // 2, (pad_num // 2 + 1)), (0, 0)))
    else:
        pad_num = item.shape[0] - item.shape[1]
        if pad_num % 2 == 0:
            padded_array = np.pad(item, pad_width=((0, 0), (pad_num // 2, pad_num // 2)))
        else:
            padded_array = np.pad(item, pad_width=((0, 0), (pad_num // 2, (pad_num // 2 + 1))))
    # padded_array = np.expand_dims(padded_array, 0).repeat(3, axis=0)
    padded_array = padded_array.astype(np.uint8)
    return Image.fromarray(padded_array)

def resize(img: Image, size=(1000, 1000), interpolation=Image.BILINEAR):
    return F.resize(img, size, interpolation)


def entrance(image_path, target_layer, save_path, model, label, device):

    # Image preprocessing
    skull_data = load_data(image_path)
    heat_path = skull_heat_map[image_path.split('/')[-1]]
    heat_data = np.load(heat_path)
    # raw_image process
    # raw_image = Image.fromarray(skull_data)
    # raw_image = raw_image.resize((1000, 1000))
    # raw_image = np.array(raw_image)
    # raw_image = raw_image[..., np.newaxis]
    # raw_image = np.repeat(raw_image, 3, axis=2)
    # raw_image = cv2.resize(raw_image, (1000,) * 2)
    # raw_image = (raw_image - np.min(raw_image)) / (np.max(raw_image) - np.min(raw_image))
    # raw_image *= 255
    # raw_image = raw_image.astype(np.uint8)

    image = transforms.Compose([
        Enhancement(),
        FixShape(),
        ToImage(),
        transforms.Resize((1000, 1000)),
        ToTensor()
    ])(skull_data)

    heat_data = transforms.Compose([
        Enhancement(),
        FixShape(),
        ToImage(),
        transforms.Resize((1000, 1000)),
        ToTensor()
    ])(heat_data)

    raw_image =  transforms.Compose([
        Enhancement(),
        FixShape(),
        ToImage(),
        transforms.Resize((1000, 1000)),
    ])(skull_data)
    raw_image = np.array(raw_image)

    raw_image = (raw_image - np.min(raw_image)) / (np.max(raw_image) - np.min(raw_image))
    raw_image *= 255
    raw_image = raw_image.astype(np.uint8)

    # image = torch.unsqueeze(image, 0)
    image = image.to(device)
    heat_data = heat_data.to(device)
    image = torch.cat([image, heat_data], dim=0)
    image = torch.unsqueeze(image, 0)

    gcam = GradCAM(model=model)
    predictions = gcam.forward(image)
    age_dir = os.path.join(save_path, str(label))
    
    pred = float(predictions.detach().cpu().numpy()[0])
    pred = round(pred, 2)
    mae = round(abs(label - pred), 2)

    # Grad-CAM
    gcam.backward()
    region = gcam.generate(target_layer=target_layer)

    file_name = image_path.split('/')[-1]
    if not os.path.exists(os.path.join(save_path, 'heat', mode, str(label))):
        os.makedirs(os.path.join(save_path, 'heat', mode, str(label)))
    if not os.path.exists(os.path.join(save_path, 'map', mode, str(label))):
        os.makedirs(os.path.join(save_path, 'map', mode, str(label)))
    if not os.path.exists(os.path.join(save_path, 'npy', mode, str(label))):
        os.makedirs(os.path.join(save_path, 'npy', mode, str(label)))
    path_heat = os.path.join(save_path, 'heat', mode, str(label), file_name.split('.')[0]+'_{}_{}.jpg'.format(str(pred), str(mae)))
    path_map = os.path.join(save_path, 'map', mode, str(label), file_name.split('.')[0]+'_{}_{}.jpg'.format(str(pred), str(mae)))
    path_npy = os.path.join(save_path, 'npy', mode, str(label), file_name.replace('.nii', '.npy'))

    save_gradcam(
        path_heat,
        path_map,
        path_npy,
        region,
        raw_image,
    )
def main(idx, mode):
    pth_path = ''   
    dataset_path = ''
    save_path = ''
    device = torch.device("cuda")
    # Model from torchvision
    use_cuda = torch.cuda.is_available()

    model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=2)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 1)
    if use_cuda:
        model = model.cuda()
    # print(model)
    latest_state = torch.load(pth_path)
    model.load_state_dict(latest_state['state_dict'])
    model.eval()

    dataset_path = os.path.join(dataset_path, '{}.csv'.format(mode))
    dataset_df = pd.read_csv(dataset_path)

    data_name_df = dataset_df['filename'].apply(lambda x: x.split('/')[-1])
    data_name_list = data_name_df.tolist()
    map_keys = list(set(skull_heat_map.keys()) & set(data_name_list))
    dataset_df = dataset_df[data_name_df.isin(map_keys)]
    file_path = dataset_df.iloc[index]['filename']
    age = file_path.split('/')[7]
    age = int(age)

    entrance(image_path=file_path,
            target_layer='_bn1', 
            save_path=save_path,
            model=model,
            label=int(age),
            device=device
            )
    print('{}:{}'.format(str(idx), file_path))
    sys.exit(0)

if __name__ == "__main__":
    index = int(sys.argv[1])
    mode = int(sys.argv[2])
    if mode == 0:
        mode = 'train'
    elif mode == 1:
        mode = 'val'
    else:
        mode = 'test'
    # mode = 'val'
    # index=0
    main(index, mode)