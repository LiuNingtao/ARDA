'''
@Author: ningtao liu
@Date: 2020-07-15 15:18:46
@LastEditors: ningtao liu
@LastEditTime: 2020-07-16 16:29:48
@FilePath: /ToothAge/ResNet/gender_main.py
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

from dataset_age import DataSetAge
from consum_transformer import Enhancement, ToTensor, ToImage
from res_18_age import ResNet18
from utils import get_performance_age

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='age', choices=['gender', 'age'])
parser.add_argument('--epoch_num', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--height', type=int, default=560)
parser.add_argument('--width', type=int, default=924)
parser.add_argument('--is_cuda', type=bool, default=True, choices=[True, False])
parser.add_argument('--data_path',
                    type=str,
                    default='/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge/DataSet/AgeDataSetMini')
parser.add_argument('--result_path', type=str, default='./result')
parser.add_argument('--save_path', type=str, default='./model_save')
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--init_lr', type=float, default=0.00005)
parser.add_argument('--lr_decay_freq', type=int, default=3)
parser.add_argument('--is_shuffle', type=eval, default=True, choices=[True, False])
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--num_class', type=int, default=9)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda_available = torch.cuda.is_available()
USE_CUDA = args.is_cuda and cuda_available

performance_train_all = {}
performance_test_all = {}


def train_model(model,
                critertion,
                train_data_loader,
                test_data_loader,
                optimizer,
                exp_lr_scheduler,
                init_epoch=0):
    if args.save_path:
        path = os.path.join(args.save_path, 'resnet_age_latest.pth')
        if os.path.exists(path):
            latest_state = torch.load(path)
            model.load_state_dict(latest_state['state_dict'])
            optimizer.load_state_dict(latest_state['optimizer'])

    loss_avg_list = []
    label_all = []
    output_all = []
    best_auc_train = -1
    best_auc_test = -1

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
            inputs, label = Variable(inputs), Variable(label)
            if USE_CUDA:
                inputs, label = inputs.cuda(), label.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = critertion(outputs, label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.cpu().data
            label_list.extend(label.cpu().data.numpy())
            output_list.extend([x.numpy() for x in outputs.cpu().data])
            count += inputs.size(0)

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, 
                        os.path.join(args.save_path, 'resnet_age_latest.pth'))
        
        performance_dict_train = get_performance_age(label_list, output_list, task='train', best_auc=best_auc_train)
        performance_train_all[epoch] = performance_dict_train
        auc_value = performance_dict_train['auc']['macro']
        if auc_value > best_auc_train:
            best_auc_train=auc_value
        loss_avg = loss_sum / count
        loss_avg_list.append(loss_avg)
        label_all.append(label_list)
        output_all.append(output_list)

        if (epoch + 1) % args.print_freq == 0:
            print('Epoch {}, Loss {}, Acc {}'.format(str(epoch),
                                                     str(loss_avg.numpy()),
                                                     str(performance_dict_train['accuracy'])))
        if (epoch + 1) % args.val_freq == 0:
            label_list, output_list, loss = test_model(model, critertion, test_data_loader)
            performance_dict_test = get_performance_age(label_list, output_list, task='test', best_auc=best_auc_test)
            performance_test_all[epoch] = performance_dict_test
            print('Epoch {}, Test Loss {}, Acc {}'.format(str(epoch),
                                                  str(loss.numpy()),
                                                  str(performance_dict_test['accuracy'])))
            auc_value = performance_dict_test['auc']['macro']
            if auc_value > best_auc_test:
                best_auc_test = auc_value
                torch.save({'epoch': epoch, 
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, 
                            os.path.join(args.save_path, 'resnet_age_best.pth'))
        

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
        inputs, label = Variable(inputs), Variable(label)
        if USE_CUDA:
            inputs, label = inputs.cuda(), label.cuda()
        outputs = model(inputs)
        loss = critertion(outputs, label)
        loss_sum += loss.cpu().data
        label_list.extend(label.cpu().data.numpy())
        output_list.extend([x.numpy() for x in outputs.cpu().data])
        count += args.batch_size
    loss_avg = loss_sum / count
    model.train()
    return label_list, output_list, loss_avg


def entrance():
    model = ResNet18()
    if USE_CUDA:
        model = model.cuda()
    print(model)
    
    critertion = nn.CrossEntropyLoss()
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

    train_dataset = DataSetAge(args.data_path, train_transform)
    test_dataset = DataSetAge(args.data_path, test_transform, mode='test')

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
        np.save(os.path.join(save_path, 'tran_per_age.pkl'), performance_train_all)
        np.save(os.path.join(save_path, 'test_per_age.pkl'), performance_test_all)
    # entrance()
