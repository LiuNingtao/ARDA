'''
@Author: ningtao liu
@Date: 2020-07-11 15:28:57
@LastEditors: ningtao liu
@LastEditTime: 2020-07-15 15:21:33
@FilePath: /ToothAge/EfficientNet/dataset.py
'''
'''
@Author: ningtao liu
@Date: 2020-07-11 15:28:57
@LastEditors: ningtao liu
@LastEditTime: 2020-07-14 16:39:14
@FilePath: /ToothAge/EfficientNet/dataset.py
'''
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
import SimpleITK as sitk
import torchvision.transforms.functional as F
import traceback
import heapq

class MyDataSet(Dataset):
    def __init__(self, root_dir:str=None, transformer=None, mode='train', task='age_re'):
        super(MyDataSet, self).__init__()
        self.root_dir = os.path.join(root_dir, mode+'.txt')
        self._get_file_list(self.root_dir)
        self.transformer = transformer
        self.task = task

    def __getitem__(self, index: int):
        file_info = self._file_list[index]
        file_path, file_label = file_info.split(',')[0].strip(), file_info.split(',')[1].strip()
        image_data = self._load_data(file_path)
        if self.transformer:
            # image_data = self.transformer(image_data)
            try:
                image_data = self.transformer(image_data)
            except Exception as exp:
                print(file_info)
                print(exp.args)
                print('='*20)
                print(traceback.format_exc())
                return {'image': torch.tensor([0]*100).long(), 'label': torch.tensor([0]).long()}
        if self.task == 'gender':
            if file_label == 'M':
                label = torch.tensor(0)
            elif file_label == 'F':
                label = torch.tensor(1)
            else:
                print('label: {} error'.format(file_label))
                return
            return {'image': image_data, 'label': label}
        elif self.task == 'age':
            age_year = int(file_label)
            if 0 <= age_year <= 4:
                label = torch.tensor(0)
            elif 5 <= age_year <= 6:
                label = torch.tensor(1)
            elif 7 <= age_year <= 9:
                label = torch.tensor(2)
            elif 10 <= age_year <= 12:
                label = torch.tensor(3)
            elif 13 <= age_year <= 15:
                label = torch.tensor(4)
            elif 16 <= age_year <= 18:
                label = torch.tensor(5)
            elif 19 <= age_year <= 21:
                label = torch.tensor(6)
            elif 22 <= age_year <= 25:
                label = torch.tensor(7)
            elif age_year > 25:
                label = torch.tensor(8)
            else:
                print('file_info: {}'.format(file_info))
                print('label: {} error'.format(age_year))
                return {'image': torch.tensor([0]*100).long(), 'label': torch.tensor([0]).long()}
            return {'image': image_data, 'label': label}
        elif self.task == 'age_re':
            label = torch.tensor(int(file_label))
            return {'image': image_data, 'label': label}
        else:
            print('error task: {}!'.format(self.task))
            raise ValueError()


    def __len__(self):
        return len(self._file_list)
    
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
    
    def _get_file_list(self, file_path):
        file_txt = open(file_path, mode='r+', encoding='utf8')
        self._file_list = file_txt.readlines()


class MyDataSetRange(Dataset):
    def __init__(self, root_dir: str = None, transformer=None, mode='train', task='age_re'):
        super(MyDataSetRange, self).__init__()
        self.root_dir = os.path.join(root_dir, mode + '.txt')
        self._get_file_list(self.root_dir)
        self.transformer = transformer
        self.task = task

    def __getitem__(self, index: int):
        file_info = self._file_list[index]
        file_path, file_label = file_info.split(',')[0].strip(), file_info.split(',')[1].strip()
        image_data = self._load_data(file_path)
        if self.transformer:
            # image_data = self.transformer(image_data)
            try:
                image_data = self.transformer(image_data)
            except Exception as exp:
                print(file_info)
                print(exp.args)
                print('=' * 20)
                print(traceback.format_exc())
                return {'image': torch.tensor([0] * 100).long(), 'label': torch.tensor([0]).long()}
        if self.task == 'gender':
            if file_label == 'M':
                label = torch.tensor(0)
            elif file_label == 'F':
                label = torch.tensor(1)
            else:
                print('label: {} error'.format(file_label))
                return
            return {'image': image_data, 'label': label}
        elif self.task == 'age':
            age_year = int(file_label)
            if 0 <= age_year <= 4:
                label = torch.tensor(0)
            elif 5 <= age_year <= 6:
                label = torch.tensor(1)
            elif 7 <= age_year <= 9:
                label = torch.tensor(2)
            elif 10 <= age_year <= 12:
                label = torch.tensor(3)
            elif 13 <= age_year <= 15:
                label = torch.tensor(4)
            elif 16 <= age_year <= 18:
                label = torch.tensor(5)
            elif 19 <= age_year <= 21:
                label = torch.tensor(6)
            elif 22 <= age_year <= 25:
                label = torch.tensor(7)
            elif age_year > 25:
                label = torch.tensor(8)
            else:
                print('file_info: {}'.format(file_info))
                print('label: {} error'.format(age_year))
                return {'image': torch.tensor([0] * 100).long(), 'label': torch.tensor([0]).long()}
            return {'image': image_data, 'label': label}
        elif self.task == 'age_re':
            label = torch.tensor(int(file_label))
            return {'image': image_data, 'label': label}
        else:
            print('error task: {}!'.format(self.task))
            raise ValueError()

    def __len__(self):
        return len(self._file_list)

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

    def _get_file_list(self, file_path):
        file_txt = open(file_path, mode='r+', encoding='utf8')
        self._file_list = file_txt.readlines()


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

# class RepeatChannel(object):
#     def __call__(self, pic):
#         array = np.array(pic)
#         array = np.repeat(array[np.newaxis, ...], 3, 0)
#         print(array.shape)
#         return array
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
