'''
Author: ningtao liu
Date: 2020-09-17 10:52:14
LastEditors: ningtao liu
LastEditTime: 2020-09-17 16:28:59
FilePath: /ToothAge/visualization/grad-cam-pytorch/customer_trans.py
'''
from cv2 import data
import numpy as np
from PIL import Image
import SimpleITK as sitk
from torchvision.transforms import functional as F
import cv2 as cv
import heapq

class FixShape(object):
    def __call__(self, item: np.ndarray):
        assert len(item.shape) == 2, print(item.shape)
        if item.shape[0] < item.shape[1]:
            pad_num = item.shape[1] - item.shape[0]
            if pad_num % 2 == 0:
                padded_array = np.pad(item, pad_width=((pad_num // 2, pad_num // 2), (0, 0)),  mode='constant')
            else:
                padded_array = np.pad(item, pad_width=((pad_num // 2, (pad_num // 2 + 1)), (0, 0)),  mode='constant')
        else:
            pad_num = item.shape[0] - item.shape[1]
            if pad_num % 2 == 0:
                padded_array = np.pad(item, pad_width=((0, 0), (pad_num // 2, pad_num // 2)), mode='constant')
            else:
                padded_array = np.pad(item, pad_width=((0, 0), (pad_num // 2, (pad_num // 2 + 1))),  mode='constant')
        # padded_array = np.expand_dims(padded_array, 0).repeat(3, axis=0)
        padded_array = padded_array.astype(np.uint8)
        return Image.fromarray(padded_array)


class Enhancement(object):
    def __call__(self, image_data: np.ndarray):
        image_data.astype(np.int16)
        image_normalized = np.zeros(image_data.shape, np.uint8)
        cv.normalize(image_data, image_normalized, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
        clahe =cv.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        dst = clahe.apply(image_normalized)
        return dst


class ToImage(object):
    def __call__(self, image_data: np.ndarray):
        if isinstance(image_data, np.ndarray):
            if len(image_data) > 2:
                image_data = np.squeeze(image_data)
            image_data = Image.fromarray(image_data)
            return image_data
        else:
            return image_data


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
        return F.to_tensor(array)

    def __repr__(self):
        return self.__class__.__name__ + '()'

