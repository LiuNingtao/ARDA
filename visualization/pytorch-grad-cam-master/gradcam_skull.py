import argparse
import cv2
import numpy as np
from numpy.lib.type_check import imag
import torch
from torch import nn
from torch.autograd import Function
from torchvision import models
import SimpleITK as sitk
import heapq
from torchvision import transforms
from customer_trans import FixShape, ToImage, Enhancement, ToTensor
from PIL import Image
import pandas as pd
import os

import sys
from efficientnet_pytorch import EfficientNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        x.register_hook(self.save_gradient)
        outputs += [x]
        # for name, module in self.model._modules.items():
        #     print('!!'*20)
        #     if name == '_avg_pooling':
        #         return outputs, x
        #     if isinstance(module, torch.nn.ModuleList) and name == '_blocks':
        #         for _, m in enumerate(module):
        #             x = m(x)
        #     else:
        #         if name == '_dropout':
        #             x = x.view(1, -1)
        #         x = module(x)
        #     if name in self.target_layers:
        #         x.register_hook(self.save_gradient)
        #         outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if isinstance(module, torch.nn.ModuleList) and name == '_blocks':
                for _, m in enumerate(module):
                    x = m(x)
            else:
                if module == self.feature_module:
                    target_activations, x = self.feature_extractor(x)
                elif "avgpool" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0),-1)
                elif name == '_dropout':
                    x = x.view(1, -1)
                else:
                    x = module(x)
        
        return target_activations, x


def preprocess_image(image_path):
    skull_data = load_data(image_path)
    inputs = transforms.Compose([
        Enhancement(),
        FixShape(),
        ToImage(),
        transforms.Resize((1000, 1000)),
        ToTensor()
    ])(skull_data)
    inputs = torch.unsqueeze(inputs, 1)

    raw_image = Image.fromarray(skull_data)
    raw_image = raw_image.resize((1000, 1000))
    raw_image = np.array(raw_image)
    raw_image = raw_image[..., np.newaxis]
    raw_image = np.repeat(raw_image, 3, axis=2)
    raw_image = cv2.resize(raw_image, (1000,) * 2)
    raw_image = (raw_image - np.min(raw_image)) / (np.max(raw_image) - np.min(raw_image))
    raw_image = raw_image.astype(np.float32)

    return inputs, raw_image

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


def show_cam_on_image(img, mask, path_heat, path_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cv2.imwrite(path_heat, np.uint8(heatmap))

    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    cv2.imwrite(path_map, np.uint8(255 * cam))

    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1,1), dtype=np.float32)
        one_hot[0][0] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, output


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1,1), dtype=np.float32)
        one_hot[0][0] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)
        a = input.grad
        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--dataset_path', type=str, default='/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge/DataSet/AgeDataSetSingleSkull/DataSetFiltered',
                        help='Input image path')
    parser.add_argument('--save_path', type=str, default='/media/gsp/LNT/DataSet/Tooth/heat-map-40/',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def entrance(index, mode):
    pth_path = r''    
    dataset_path = r''
    save_path = r''
    use_cuda = torch.cuda.is_available()

    model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 1)
    model.eval()
    if use_cuda:
        model = model.cuda()
    # print(model)
    latest_state = torch.load(pth_path)
    model.load_state_dict(latest_state['state_dict'])

    grad_cam = GradCam(model=model, feature_module=model._conv_head, \
                       target_layer_names=["_conv_head"], use_cuda=use_cuda)
    dataset_path = os.path.join(dataset_path, '{}.csv'.format(mode))

    dataset_df = pd.read_csv(dataset_path)
    file_path = dataset_df.iloc[index]['filename']
    age, file_name = file_path.split('/')[7], file_path.split('/')[-1]
    age = int(age)
    inputs, img = preprocess_image(file_path)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask, output = grad_cam(inputs, target_index)
    output = round(float(output), 2)
    ae = round(abs(age - output), 2)

    if not os.path.exists(os.path.join(save_path, 'heat', mode, str(age))):
        os.makedirs(os.path.join(save_path, 'heat', mode, str(age)))
    if not os.path.exists(os.path.join(save_path, 'map', mode, str(age))):
        os.makedirs(os.path.join(save_path, 'map', mode, str(age)))
    path_heat = os.path.join(save_path, 'heat', mode, str(age), file_name.split('.')[0]+'_{}_{}.jpg'.format(str(output), str(ae)))
    path_map = os.path.join(save_path, 'map', mode, str(age), file_name.split('.')[0]+'_{}_{}.jpg'.format(str(output), str(ae)))

    show_cam_on_image(img, mask, path_heat, path_map)
    print('{}:{}'.format(str(index), file_name))

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # print(model._modules.items())
    # gb = gb_model(inputs, index=target_index)
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask*gb)
    # gb = deprocess_image(gb)

    # cv2.imwrite('gb.jpg', gb)
    # cv2.imwrite('cam_gb.jpg', cam_gb)

if __name__ == '__main__':
    # index = int(sys.argv[1])
    # mode = int(sys.argv[2])
    # if mode == 0:
    #     mode = 'train'
    # elif mode == 1:
    #     mode = 'val'
    # else:
    #     mode = 'test'

    mode = 'train'
    index = 1
    entrance(index, mode)