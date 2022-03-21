from cv2 import data
import numpy as np
from PIL import Image
import SimpleITK as sitk
from torchvision.transforms import functional as F
import cv2 as cv
import heapq


def img_loader(path):
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

    def __call__(self, image_data:np.ndarray):
        _range = np.max(image_data) - np.min(image_data)
        return (image_data - np.min(data)) / _range


class Standardization(object):
    def __call__(self, image_data:np.ndarray):
        mu = np.mean(image_data)
        sigma = np.std(image_data)
        return (data - mu) / sigma


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


class FixShape(object):
    def __call__(self, item: np.ndarray):
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


if __name__ == '__main__':
    path = r''
    img_loader(path)
