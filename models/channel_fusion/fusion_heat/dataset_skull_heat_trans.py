from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
import SimpleITK as sitk
import torchvision.transforms.functional as F
import traceback
import heapq
import cv2
from efficientnet_pytorch import EfficientNet
from torch import nn
import cv2

import sys
sys.path.append(r'/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge/visualization/grad-cam-pytorch')
from grad_cam_skull import GradCAM


class DataSetAgeFusion(Dataset):
    """
    skull+heat map
    分别进行预处理并且进行数据增广
    """
    def __init__(self, data_df, heat_dir, transformer_skull, transformer_heat, mode='train', fusion_mode='channel'):
        super(DataSetAgeFusion, self).__init__()
        self.data_df = data_df
        self.mode = mode
        self.transformer_skull = transformer_skull
        self.transformer_heat = transformer_heat
        self.fusion_mode = fusion_mode
        self.skull_heat_map = self._get_img_heat_map(heat_dir)
        self.heat_age_list = [age for age in range(4, 41)]
        # self.heat_age_list = [4, 6, 9, 24, 27, 29, 30, 32, 33, 35, 36, 38, 39, 40]
    

    def _get_img_heat_map(self, heat_dir):
        file_path_list = []
        for root, _, file_list in os.walk(heat_dir):
            for file_name in file_list:
                file_path_list.append(os.path.join(root, file_name))
        img_heat_map = dict(zip(list(map(lambda file_name: '_'.join(file_name.split('/')[-1].split('.')[0].split('_')[0:-1])+'.nii', file_path_list)), file_path_list))
        map_keys = list(img_heat_map.keys())
        data_name_df = self.data_df['filename'].apply(lambda x: x.split('/')[-1])
        data_name_list = data_name_df.tolist()
        map_keys = list(set(map_keys) & set(data_name_list))
        self.data_df = self.data_df[data_name_df.isin(map_keys)]
        return img_heat_map

    def __getitem__(self, index: int):
        try:
            skull_path = self.data_df.iloc[index]['filename']
            skull_name = skull_path.split('/')[-1]
            heat_path = self.skull_heat_map[skull_name]
        except:
            print('except key error')
            return  {'image': torch.ones(2, 1000, 1000)*(-10000.0), 'label': torch.tensor(0.0)}
        skull_data = self._load_data(skull_path)
        heat_data = self._load_heat(heat_path)

        if self.transformer_skull:
            skull_data = self.transformer_skull(skull_data)
        
        if self.transformer_heat:
            heat_data = self.transformer_heat(heat_data)

        assert skull_data.shape == heat_data.shape

        age = int(skull_path.split('/')[7])

        age = torch.tensor(age).float()

        if self.fusion_mode == 'channel':
            # if age in self.heat_age_list or self.mode == 'train' or self.mode == 'val':
            #     image_data = torch.cat([skull_data, heat_data], dim=0)
            # else:
            #     image_data = torch.cat([skull_data, skull_data], dim=0)
            image_data = torch.cat([skull_data, heat_data], dim=0)
            return {'image': image_data, 'label': age}
        else:
            return {'skull': skull_data, 'heat': heat_data, 'label': age}
        

    def __len__(self):
        return len(self.data_df)
    
    def _load_data(self, path):
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

    def _load_heat(self, heat_path):
        heat_data = np.load(heat_path)
        return heat_data
    

class DataSetMiniCAM(Dataset):
    """
    skull+heat map
    分别进行预处理并且进行数据增广
    """
    def __init__(self, data_df, pretrain_path, transformer_skull, transformer_heat, mode='train', fusion_mode='channel'):
        super(DataSetMiniCAM, self).__init__()
        self.data_df = data_df
        self.mode = mode
        self.transformer_skull = transformer_skull
        self.transformer_heat = transformer_heat
        self.fusion_mode = fusion_mode
        self.heat_age_list = [age for age in range(4, 41)]

        model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1)
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, 1)
        model = model.cuda()
        # print(model)
        latest_state = torch.load(pretrain_path)
        model.load_state_dict(latest_state['state_dict'])
        model.eval()
        gradCAM = GradCAM(model=model)
        self.gcam = gradCAM
        self.heat_npy_path = r'/media/gsp/LNT/DataSet/Tooth/heat-map/heat_map_mini_CAM/npy/'
        # self.heat_age_list = [4, 6, 9, 24, 27, 29, 30, 32, 33, 35, 36, 38, 39, 40]
    

    def _get_img_heat_map(self, image, image_path):
        save_path = os.path.join(self.heat_npy_path, self.mode, image_path.split('/')[7], image_path.split('/')[-1].replace('nii', 'npy'))
        if not os.path.exists(save_path):
            if not os.path.exists(os.path.join(self.heat_npy_path, self.mode, image_path.split('/')[7])):
                os.makedirs(os.path.join(self.heat_npy_path, self.mode, image_path.split('/')[7]))
            image = torch.unsqueeze(image, 0)
            image = image.cuda()
            _ = self.gcam.forward(image)      
            # Grad-CAM
            self.gcam.backward()
            region = self.gcam.generate(target_layer='_bn1')
            saliency_map = cv2.resize(region, (1000, 1000))
            np.save(os.path.join(self.heat_npy_path, self.mode, image_path.split('/')[7], image_path.split('/')[-1].replace('nii', 'npy')), saliency_map)
        else:
            saliency_map = np.load(save_path)
        return saliency_map
        # file_path_list = []
        # for root, _, file_list in os.walk(heat_dir):
        #     for file_name in file_list:
        #         file_path_list.append(os.path.join(root, file_name))
        # img_heat_map = dict(zip(list(map(lambda file_name: '_'.join(file_name.split('/')[-1].split('.')[0].split('_')[0:-1])+'.nii', file_path_list)), file_path_list))
        # map_keys = list(img_heat_map.keys())
        # data_name_df = self.data_df['filename'].apply(lambda x: x.split('/')[-1])
        # data_name_list = data_name_df.tolist()
        # map_keys = list(set(map_keys) & set(data_name_list))
        # self.data_df = self.data_df[data_name_df.isin(map_keys)]
        # return img_heat_map

    def __getitem__(self, index: int):
        skull_path = self.data_df.iloc[index]['filename']
        skull_data = self._load_data(skull_path)

        if self.transformer_skull:
            skull_data = self.transformer_skull(skull_data)

        age = int(skull_path.split('/')[7])
        
        heat_data = self._get_img_heat_map(skull_data, skull_path)
        if self.transformer_heat:
            heat_data = self.transformer_heat(heat_data)

        assert skull_data.shape == heat_data.shape


        age = torch.tensor(age).float()

        if self.fusion_mode == 'channel':
            # if age in self.heat_age_list or self.mode == 'train' or self.mode == 'val':
            #     image_data = torch.cat([skull_data, heat_data], dim=0)
            # else:
            #     image_data = torch.cat([skull_data, skull_data], dim=0)
            image_data = torch.cat([skull_data, heat_data], dim=0)
            return {'image': image_data, 'label': age}
        else:
            return {'skull': skull_data, 'heat': heat_data, 'label': age}
        

    def __len__(self):
        return len(self.data_df)
    
    def _load_data(self, path):
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




class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        array = np.array(pic)
        array = np.expand_dims(array, 2).repeat(3, axis=2)
        return F.to_tensor(array)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class FixShape(object):
    def __call__(self, item: np.ndarray):
        assert len(item.shape) == 2, print(item.shape)
        # if item.shape[0] < item.shape[1]:
        #     pad_num = item.shape[1] - item.shape[0]
        #     if pad_num % 2 == 0:
        #         padded_array = np.pad(item, pad_width=((pad_num // 2, pad_num // 2), (0, 0)))
        #     else:
        #         padded_array = np.pad(item, pad_width=((pad_num // 2, (pad_num // 2 + 1)), (0, 0)))
        # else:
        #     pad_num = item.shape[0] - item.shape[1]
        #     if pad_num % 2 == 0:
        #         padded_array = np.pad(item, pad_width=((0, 0), (pad_num // 2, pad_num // 2)))
        #     else:
        #         padded_array = np.pad(item, pad_width=((0, 0), (pad_num // 2, (pad_num // 2 + 1))))
        # # padded_array = np.expand_dims(padded_array, 0).repeat(3, axis=0)
        # padded_array = padded_array.astype(np.uint8)
        return Image.fromarray(item.astype(np.uint8))
