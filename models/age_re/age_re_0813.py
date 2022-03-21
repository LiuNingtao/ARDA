'''
@Author: ningtao liu
@Date: 2020-07-15 15:18:46
@LastEditors: ningtao liu
@LastEditTime: 2020-07-16 16:29:48
@FilePath: /ToothAge/ResNet/gender_main.py
年龄大于25岁的回归
'''
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import os
from tqdm import tqdm
from torch import nn
import numpy as np
import traceback
import sys
import pandas as pd

from efficientnet_pytorch import EfficientNet
from dataset import MyDataSetRange as MyDataSet
from consum_transformer import Enhancement, ToTensor, ToImage
from res_18_age_re import ResNet18
from utils import get_performance_age_range

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='age_re', choices=['gender', 'age', 'age_re'])
parser.add_argument('--epoch_num', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--height', type=int, default=450)
parser.add_argument('--width', type=int, default=900)
parser.add_argument('--is_cuda', type=bool, default=True, choices=[True, False])
parser.add_argument('--data_path',
                    type=str,
                    default='/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge/DataSet/AgeDataSetUper3')
parser.add_argument('--result_path', type=str, default='./result_0811')
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--init_lr', type=float, default=0.00005)
parser.add_argument('--lr_decay_freq', type=int, default=3)
parser.add_argument('--is_shuffle', type=eval, default=True, choices=[True, False])
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--num_class', type=int, default=1)
parser.add_argument('--is_pertrain', type=eval, default=False, choices=[True, False])
parser.add_argument('--model', type=str, default='efficient-b0')
parser.add_argument('--desc', type=str, default='3-80岁回归，仅JAW')
parser.add_argument('--trick', type=str, default='None')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda_available = torch.cuda.is_available()
USE_CUDA = args.is_cuda and cuda_available

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
porfile_path= os.path.join(args.result_path, 'profile.txt')
profile_file =  open(porfile_path, 'w+', encoding='utf8')
print(args, file=profile_file, flush=True)
profile_file.close()

def train_model(model,
                critertion,
                train_data_loader,
                test_data_loader,
                optimizer,
                exp_lr_scheduler,
                init_epoch=0):
    if args.result_path:
        path = os.path.join(args.result_path, 'model_save', 'model_train_latest.pth')
        if os.path.exists(path):
            latest_state = torch.load(path)
            model.load_state_dict(latest_state['state_dict'])
            optimizer.load_state_dict(latest_state['optimizer'])
            print('='*20+'load old'+'='*20)

    min_loss_train = sys.maxsize
    min_loss_test = sys.maxsize
    train_performance = {'epoch': [], 'loss': [], 'acc_0': [], 'acc_1': [], 'acc_2': []}
    test_performance = {'epoch': [], 'loss': [], 'acc_0': [], 'acc_1': [], 'acc_2': []}

    model.train()
    for epoch in range(init_epoch, args.epoch_num):
        loss_sum = 0
        count = 0
        label_list = []
        output_list = []
        optimizer = exp_lr_scheduler(optimizer, epoch)
        for data in tqdm(train_data_loader):
            inputs, label = data['image'], data['label']
            if inputs.size(1) != 1:
                continue
            inputs = torch.cat([inputs for i in range(3)], dim=1)
            # print(inputs.size())
            label = label.float()
            inputs, label = Variable(inputs), Variable(label)
            if USE_CUDA:
                inputs, label = inputs.cuda(), label.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = critertion(torch.squeeze(outputs), torch.squeeze(label))
            loss.backward()
            optimizer.step()

            loss_sum += loss.cpu().data.numpy()
            label_list.extend(label.cpu().data.numpy())
            output_list.extend([x.numpy() for x in outputs.cpu().data])
            count += 1
        
        model_save_path = os.path.join(args.result_path, 'model_save')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, 
                    os.path.join(model_save_path, 'model_train_latest.pth'))
        loss_avg = loss_sum / count

        result_save_path = os.path.join(args.result_path, 'result')
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        
        np.save(os.path.join(result_save_path, 'label_{}_train'.format(str(epoch))), np.array(label_list))
        np.save(os.path.join(result_save_path, 'output_{}_train'.format(str(epoch))), np.array(output_list))
        acc_0, acc_1, acc_2 = get_performance_age_range(np.array(label_list), np.array(output_list))

        performance_path = os.path.join(args.result_path, 'performance')
        if not os.path.exists(performance_path):
            os.makedirs(performance_path)
        train_performance['epoch'].append(epoch)
        train_performance['loss'].append(loss_avg)
        train_performance['acc_0'].append(acc_0)
        train_performance['acc_1'].append(acc_1)
        train_performance['acc_2'].append(acc_2)
        pd.DataFrame(train_performance).to_csv(os.path.join(performance_path, 'train.csv'), index=False)

        if (epoch + 1) % args.print_freq == 0:
            print('-'*80)
            print('task: {}, epoch: {}, data_num: {}, loss_avg: {}'.format('train', str(epoch), str(count * args.batch_size), str(loss_avg)))
            print('acc_0: {}, acc_1: {}, acc_2: {}'.format(str(acc_0), str(acc_1), str(acc_2)))

        if loss_avg < min_loss_train:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        },
                        os.path.join(model_save_path, 'model_train_best.pth'))
            min_loss_train = loss_avg
            print('='*10 + 'train model saved' + '='*10)

        if (epoch + 1) % args.val_freq == 0:
            label_list_test, output_list_test, loss_avg_test, count_test = test_model(model, critertion, test_data_loader)
            acc_0_test, acc_1_test, acc_2_test = get_performance_age_range(np.array(label_list_test), np.array(output_list_test))

            np.save(os.path.join(result_save_path, 'label_{}_test'.format(str(epoch))), np.array(label_list_test))
            np.save(os.path.join(result_save_path, 'output_{}_test'.format(str(epoch))), np.array(output_list_test))

            test_performance['epoch'].append(epoch)
            test_performance['loss'].append(loss_avg_test)
            test_performance['acc_0'].append(acc_0_test)
            test_performance['acc_1'].append(acc_1_test)
            test_performance['acc_2'].append(acc_2_test)

            pd.DataFrame(test_performance).to_csv(os.path.join(performance_path, 'test.csv'), index=False)
            test_batch_size = max((args.batch_size // 2), 1)
            print('-'*80)
            print('task: {}, epoch: {}, data_num: {}, loss_avg: {}'.format('test', str(epoch), str(count_test * test_batch_size), str(loss_avg_test)))
            print('acc_0: {}, acc_1: {}, acc_2: {}'.format(str(acc_0_test), str(acc_1_test), str(acc_2_test)))

            if loss_avg_test < min_loss_test:
                torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, 
                        os.path.join(model_save_path, 'model_test_best.pth'))
                min_loss_test = loss_avg_test
                print('='*10 + 'test model saved' + '='*10)
        

def test_model(model, critertion, dataloader):
    model.eval()
    loss_sum = 0
    count = 0
    label_list = []
    output_list = []
    torch.cuda.empty_cache()
    for data in tqdm(dataloader):
        inputs, label = data['image'], data['label']
        if inputs.size(1) != 1:
            continue
        inputs = torch.cat([inputs for i in range(3)], dim=1)
        label = label.float()

        inputs, label = Variable(inputs), Variable(label)
        if USE_CUDA:
            inputs, label = inputs.cuda(), label.cuda()
        outputs = model(inputs)
        loss = critertion(torch.squeeze(outputs), np.squeeze(label))
        loss_sum += loss.cpu().data.numpy()
        label_list.extend(label.cpu().data.numpy())
        output_list.extend([x.numpy() for x in outputs.cpu().data])
        count += 1
    loss_avg = loss_sum / count
    model.train()
    return label_list, output_list, loss_avg, count


def entrance():
    # model = ResNet18()
    # if USE_CUDA:
    #     model = model.cuda()
    if args.is_pertrain:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    else:
        model = EfficientNet.from_name('efficientnet-b0')

    # net_weight_path = '/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/Resource/efficientnet-b0-355c32eb.pth'
    # state_dict = torch.load(net_weight_path)
    # model.load_state_dict(state_dict)

    # 修改网络结构
    # model._conv_stem.in_channels = 1
    # model._conv_stem.weight = torch.nn.Parameter(torch.cat([model._conv_stem.weight, model._conv_stem.weight], axis=1))
    # model._conv_stem.weight = torch.nn.Parameter(model._conv_stem.weight[:,0, :, :])
    
    # model._conv_stem.stride = torch.nn.Parameter(model._conv_stem.stride[0])


    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 1)
    if USE_CUDA:
        model = model.cuda()
    # print(model)
    
    critertion = nn.L1Loss()
    if USE_CUDA:
        critertion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=0.00005)

    train_transform = transforms.Compose([
        Enhancement(),
        ToImage(),
        transforms.Resize((args.height, args.width)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        ToTensor()
    ])

    test_transform = transforms.Compose([
        Enhancement(),
        ToImage(),
        transforms.Resize((args.height, args.width)),
        ToTensor()
    ])

    train_dataset = MyDataSet(args.data_path, train_transform, task=args.task)
    test_dataset = MyDataSet(args.data_path, test_transform, mode='test', task=args.task)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.is_shuffle,
                                  num_workers=args.num_works,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=max((args.batch_size // 2), 1),
                                 shuffle=args.is_shuffle,
                                 num_workers=args.num_works,
                                 drop_last=True)
    train_model(model, critertion, train_dataloader, test_dataloader, optimizer, exp_lr_scheduler)


def exp_lr_scheduler(optimizer, epoch, init_lr=args.init_lr, lr_decay_epoch=args.lr_decay_freq):
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))                                        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer


if __name__ == '__main__':
    try:
        entrance()
    except Exception as exp:
        print(exp.args)
        print('=' * 20)
        print(traceback.format_exc())
    finally:
        save_path = './result'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    # entrance()
