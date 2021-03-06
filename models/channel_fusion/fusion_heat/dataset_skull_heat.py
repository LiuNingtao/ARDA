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
from PIL import Image


class DataSetAgeFusion(Dataset):
    def __init__(self, data_df, heat_dir, transformer, mode='train', fusion_mode='heat'):
        super(DataSetAgeFusion, self).__init__()
        self.data_df = data_df
        self.mode = mode
        self.transformer = transformer
        self.fusion_mode = fusion_mode
        self.skull_heat_map = self._get_img_heat_map(heat_dir)

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
        heat_data = self.get_heat(heat_path)
        skull_data = self.get_skull(skull_path)
        
        assert heat_data.shape == skull_data.shape
        image_data =np.array([heat_data, skull_data])
        image_data = np.float32(image_data)

        if self.transformer:
            image_data = self.transformer(image_data)
        age = int(skull_path.split('/')[7])
        age = torch.tensor(age).float()
        if self.fusion_mode == 'heat':
            return {'image': torch.tensor(image_data), 'label': age}
        else:
            NotImplemented

 

    def __len__(self):
        return len(self.data_df)
    

    def get_skull(self, img_path, size=(1000, 1000)):
        img_data = self._load_data(img_path)
        img_data = self._enhancement(img_data)
        img_data = self._fix_shape(img_data)
        img_data = self._resize(img_data, size)
        img_data = img_data / np.max(img_data)
        return img_data
    
    def get_heat(self, heat_path):
        heat_data = self._load_heat(heat_path)
        heat_data = heat_data / np.max(heat_data)
        return heat_data

    
    def _load_data(self, path):
        image = sitk.ReadImage(path)
        image_data = sitk.GetArrayFromImage(image)
        image_data = np.squeeze(image_data)
        # if len(image_data.shape) != 3:
        #     image_data = np.squeeze(image_data)
        #     image_data = image_data[np.newaxis, :, :]
        if len(image_data.shape) > 2:
            shape_list = image_data.shape
            # ?????????2???????????????????????????nsmallest?????????????????????????????????
            max_index = map(shape_list.index, heapq.nlargest(2, shape_list))
            # max_index ?????????????????????????????????list()??????set()????????????
            max_index = set(max_index)
            if max_index == {0, 1}:
                image_data = image_data[..., 0]
            elif max_index == {1, 2}:
                image_data = image_data[0, ...]
        return image_data


    def _load_heat(self, heat_path):
        heat_data = np.array(cv2.imread(heat_path, cv2.IMREAD_GRAYSCALE))
        return heat_data
    

    def _enhancement(self, image_data:np.ndarray):
        image_data.astype(np.int16)
        image_normalized = np.zeros(image_data.shape, np.uint8)
        a, b =  np.min(image_data), np.max(image_data)
        cv2.normalize(image_data, image_normalized, np.min(image_data), np.max(image_data), cv2.NORM_MINMAX, cv2.CV_8U)
        c, d =  np.min(image_normalized), np.max(image_normalized)
        clahe =cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        dst = clahe.apply(image_data)
        return dst
    

    def _fix_shape(self, img_data):
        if img_data.shape[0] < img_data.shape[1]:
            pad_num = img_data.shape[1] - img_data.shape[0]
            if pad_num % 2 == 0:
                padded_array = np.pad(img_data, pad_width=((pad_num // 2, pad_num // 2), (0, 0)))
            else:
                padded_array = np.pad(img_data, pad_width=((pad_num // 2, (pad_num // 2 + 1)), (0, 0)))
        else:
            pad_num = img_data.shape[0] - img_data.shape[1]
            if pad_num % 2 == 0:
                padded_array = np.pad(img_data, pad_width=((0, 0), (pad_num // 2, pad_num // 2)))
            else:
                padded_array = np.pad(img_data, pad_width=((0, 0), (pad_num // 2, (pad_num // 2 + 1))))
        # padded_array = np.expand_dims(padded_array, 0).repeat(3, axis=0)
        return padded_array.astype(np.float32)

    def _resize(self, img_data: np.ndarray, size:tuple):
        img_data = Image.fromarray(img_data)
        return np.array(F.resize(img_data, size, Image.BILINEAR))
    


    




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
