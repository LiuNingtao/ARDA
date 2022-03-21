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


class DataSetAgeFusion(Dataset):
    """
    skull+heat map
    分别进行预处理并且进行数据增广
    """
    def __init__(self, data_df, data_dir, transformer, mode='train', fusion_mode='channel', fusion_list=['tooth']):
        super(DataSetAgeFusion, self).__init__()
        self.data_df = data_df
        self.data_dir = data_dir
        self.mode = mode
        self.transformer = transformer
        self.fusion_list = fusion_list
        self.fusion_mode = fusion_mode
    
    def __getitem__(self, index: int):
        skull_path = self.data_df.iloc[index]['filename']
        age, skull_name = skull_path.split('/')[-2], skull_path.split('/')[-1]
        part_name_list = [skull_name.split('.')[0]+'_'+x+'.nii' for x in self.fusion_list]
        part_path_list = [os.path.join(self.data_dir, age, x) for x in part_name_list]
        part_data_list = []

        for path in part_path_list:
            part_data = self._load_data(path)
            if self.transformer:
                part_data = self.transformer(part_data)
            part_data_list.append(part_data)

        age = torch.tensor(float(age)).float()

        if self.fusion_mode == 'channel':
            image_data = torch.cat(part_data_list, dim=0)
            return {'image': image_data, 'label': age}
        else:
            return dict(zip(self.fusion_list+['label'], part_data_list+[age]))
        

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
        heat_data = np.array(cv2.imread(heat_path, cv2.IMREAD_GRAYSCALE))
        return heat_data
    




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
