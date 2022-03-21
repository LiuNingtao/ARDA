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
from sklearn.model_selection import train_test_split

from efficientnet_pytorch import EfficientNet
from dataset_skull_part import DataSetAgeFusion as MyDataSet

sys.path.append(r'/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge')
from consum_transformer import Enhancement, ToTensor, ToImage, FixShape
from performance_tools import get_performance_age_range


parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='age_re', choices=['gender', 'age', 'age_re'])
parser.add_argument('--epoch_num', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--height', type=int, default=500)
parser.add_argument('--width', type=int, default=500)
parser.add_argument('--is_cuda', type=bool, default=True, choices=[True, False])

parser.add_argument('--data_dir',
                    type=str,
                    default='/media/gsp/LNT/DataSet/Tooth/SKULL_cutted/')
parser.add_argument('--dataset_path',
                    type=str,
                    default='/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge/DataSet/AgeDataSetSingleSkull/DataSetFiltered')        

parser.add_argument('--result_path', type=str, default='/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge/models/channel_fusion/fusion_part/result/1115_top_vertebra')
parser.add_argument('--load_path', type=str, default='/media/gsp/48cfceb8-8b77-4141-bba7-da05abd58d95/2019/lnt/project/ToothAge/models/channel_fusion/fusion_part/result/1115_top_vertebra')
parser.add_argument('--load_old', type=bool, default=True, choices=[True, False])

parser.add_argument('--is_top', type=bool, default=True, choices=[True, False])
parser.add_argument('--is_vertebra', type=bool, default=True, choices=[True, False])
parser.add_argument('--is_tooth', type=bool, default=False, choices=[True, False])

parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--init_lr', type=float, default=0.0001)
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--lr_decay_freq', type=int, default=4)
parser.add_argument('--is_shuffle', type=eval, default=True, choices=[True, False])
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--num_class', type=int, default=1)
parser.add_argument('--is_pertrain', type=eval, default=True, choices=[True, False])
parser.add_argument('--model', type=str, default='efficient-b0')
parser.add_argument('--fusion_mode', type=str, default='channel', choices=['channel', 'model', 'head'])
parser.add_argument('--desc', type=str, default='各个部分　Top+vertbra')
parser.add_argument('--trick', type=str, default='None')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
USE_CUDA = args.is_cuda and cuda_available

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
porfile_path= os.path.join(args.result_path, 'profile.txt')
profile_file =  open(porfile_path, 'w+', encoding='utf8')
print(args, file=profile_file, flush=True)
profile_file.close()

result_path = os.path.join(args.result_path, 'result')
if not os.path.exists(result_path):
    os.makedirs(result_path)

model_path = os.path.join(args.result_path, 'model_save')
if not os.path.exists(model_path):
    os.makedirs(model_path)

per_path = os.path.join(args.result_path, 'performance')
if not os.path.exists(per_path):
    os.makedirs(per_path)

pic_path = os.path.join(args.result_path, 'pic')
if not os.path.exists(pic_path):
    os.makedirs(pic_path)

def train_model(model,
                critertion,
                train_data_loader,
                test_data_loader,
                val_data_loader,
                optimizer,
                exp_lr_scheduler,
                init_epoch=16):

    init_epoch = 0
    if args.load_old:
        if args.result_path:
            path = os.path.join(args.load_path, 'model_save', 'model_train_latest.pth')
            if os.path.exists(path):
                latest_state = torch.load(path)
                init_epoch = int(latest_state['epoch'])+1
                model.load_state_dict(latest_state['state_dict'])
                optimizer.load_state_dict(latest_state['optimizer'])
                print('='*60+'load old'+'='*60)
    
    train_per_path = os.path.join(per_path, 'train.csv')
    val_per_path = os.path.join(per_path, 'val.csv')
    test_per_path = os.path.join(per_path, 'test.csv')
    if not os.path.exists(train_per_path):
        train_performance = {'epoch': [], 'loss': []}

    else:
        train_performance = pd.read_csv(train_per_path).to_dict(orient='list')

    if not os.path.exists(val_per_path):
        val_performance = {'epoch': [], 'loss': []}
        
    else:
        val_performance = pd.read_csv(val_per_path).to_dict(orient='list')

    if not os.path.exists(test_per_path):
        test_performance = {'epoch': [], 'loss': []}
        
    else:
        test_performance = pd.read_csv(test_per_path).to_dict(orient='list')
    
    min_loss_train = min(train_performance['loss'] + [sys.maxsize])
    min_loss_val = min(val_performance['loss'] + [sys.maxsize])
    min_loss_test = min(test_performance['loss'] + [sys.maxsize])

    model.train()
    for epoch in range(init_epoch, args.epoch_num):
        loss_sum = 0
        count = 0
        label_list = []
        output_list = []
        optimizer = exp_lr_scheduler(optimizer, epoch)
        for data in tqdm(train_data_loader):
            inputs, label = data['image'], data['label']
            # inputs = torch.cat([inputs for i in range(3)], dim=1)
            # print(inputs.size())
            if torch.min(inputs)<=-10000:
                print('path error')
                continue
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
        
        performance_path = os.path.join(args.result_path, 'performance')
        if not os.path.exists(performance_path):
            os.makedirs(performance_path)
        train_performance['epoch'].append(epoch)
        train_performance['loss'].append(loss_avg)

        pd.DataFrame(train_performance).to_csv(os.path.join(performance_path, 'train.csv'), index=False)

        if (epoch + 1) % args.print_freq == 0:
            print('-'*80)
            print('task: {}, epoch: {}, data_num: {}, loss_avg: {}'.format('train', str(epoch), str(count * args.batch_size), str(loss_avg)))

        if loss_avg < min_loss_train:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        },
                        os.path.join(model_save_path, 'model_train_best.pth'))
            min_loss_train = loss_avg
            np.save(os.path.join(result_save_path, 'label_best_train'), np.array(label_list))
            np.save(os.path.join(result_save_path, 'output_best_train'), np.array(output_list))
            print('='*10 + 'train model saved' + '='*10)

        is_test = False
        if (epoch) % args.val_freq == 0:
            label_list_val, output_list_val, loss_avg_val, count_val = test_model(model, critertion, val_data_loader)

            val_performance['epoch'].append(epoch)
            val_performance['loss'].append(loss_avg_val)

            pd.DataFrame(val_performance).to_csv(os.path.join(performance_path, 'val.csv'), index=False)
            val_batch_size = max((args.batch_size // 2), 1)
            print('-'*80)
            print('task: {}, epoch: {}, data_num: {}, loss_avg: {}'.format('val', str(epoch), str(count_val * val_batch_size), str(loss_avg_val)))

            if loss_avg_val < min_loss_val:
                is_test = True
                torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, 
                        os.path.join(model_save_path, 'model_val_best.pth'))
                min_loss_val = loss_avg_val
                np.save(os.path.join(result_save_path, 'label_best_val'), np.array(label_list_val))
                np.save(os.path.join(result_save_path, 'output_best_val'), np.array(output_list_val))
                print('='*10 + 'val model saved' + '='*10)

        if is_test:
            label_list_test, output_list_test, loss_avg_test, count_test = test_model(model, critertion, test_data_loader)

            test_performance['epoch'].append(epoch)
            test_performance['loss'].append(loss_avg_test)

            pd.DataFrame(test_performance).to_csv(os.path.join(performance_path, 'test.csv'), index=False)
            test_batch_size = max((args.batch_size // 2), 1)
            print('-'*80)
            print('task: {}, epoch: {}, data_num: {}, loss_avg: {}'.format('test', str(epoch), str(count_test * test_batch_size), str(loss_avg_test)))

            if loss_avg_test < min_loss_test:
                torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, 
                        os.path.join(model_save_path, 'model_test_best.pth'))
                min_loss_test = loss_avg_test
                
                np.save(os.path.join(result_save_path, 'label_best_test'), np.array(label_list_test))
                np.save(os.path.join(result_save_path, 'output_best_test'), np.array(output_list_test))

                print('='*10 + 'test model saved' + '='*10)
                print('='*10 + 'new best' + '='*10)
        

def test_model(model, critertion, dataloader):
    model.eval()
    loss_sum = 0
    count = 0
    label_list = []
    output_list = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, label = data['image'], data['label']
            # inputs = torch.cat([inputs for i in range(3)], dim=1)
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

    part_list = []
    if args.is_tooth:
        part_list.append('tooth')
    if args.is_top:
        part_list.append('top')
    if args.is_vertebra:
        part_list.append('vertebra')
    
    in_channels =len(part_list)

    if args.is_pertrain:
        model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=in_channels)
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
        FixShape(),
        ToImage(),
        transforms.Resize((args.height, args.width)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        ToTensor()
    ])


    test_transform = transforms.Compose([
        Enhancement(),
        FixShape(),
        ToImage(),
        transforms.Resize((args.height, args.width)),
        ToTensor()
    ])

    train_df = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.dataset_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(args.dataset_path, 'test.csv'))


    train_dataset = MyDataSet(train_df, args.data_dir, train_transform, mode='train', fusion_mode=args.fusion_mode, fusion_list=part_list)
    test_dataset = MyDataSet(test_df, args.data_dir, test_transform, mode='test', fusion_mode=args.fusion_mode, fusion_list=part_list)
    val_dataset = MyDataSet(val_df, args.data_dir, test_transform, mode='val', fusion_mode=args.fusion_mode, fusion_list=part_list)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.is_shuffle,
                                  num_workers=args.num_works,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                 batch_size=max((args.batch_size // 2), 1),
                                 shuffle=args.is_shuffle,
                                 num_workers=args.num_works,
                                 drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=max((args.batch_size // 2), 1),
                                 shuffle=args.is_shuffle,
                                 num_workers=args.num_works,
                                 drop_last=True)
    train_model(model, critertion, train_dataloader, val_dataloader, test_dataloader, optimizer, exp_lr_scheduler)


def exp_lr_scheduler(optimizer, epoch, init_lr=args.init_lr, lr_decay_epoch=args.lr_decay_freq):
    lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))                                        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer


if __name__ == '__main__':
    # try:
    #     entrance()
    # except Exception as exp:
    #     print(exp.args)
    #     print('=' * 20)
    #     print(traceback.format_exc())
    # finally:
    #     save_path = './result'
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    entrance()
