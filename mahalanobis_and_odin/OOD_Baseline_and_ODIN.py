"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
from torchvision import models
import os
import lib_generation
from torchvision import transforms
from torch.autograd import Variable
import sys
sys.path.append('../')
from my_models import densenet_121, mobilenet, resnet_50, vgg_16

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default="skin_cancer")
parser.add_argument('--dataroot', default='../data', help='path to datasets')
parser.add_argument('--outf', default='./output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=8, help='the # of classes')
parser.add_argument('--net_type', default="resnet_50", help='densenet_121|mobilenet|resnet_50|vgg_16')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)

def main():
    # set the path to pre-trained model and output
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    
    out_dist_list = ['skin_cli', 'skin_derm', 'corrupted', 'corrupted_70', 'imgnet', 'nct']
    
    # load networks
    if args.net_type == 'densenet_121':
        model = densenet_121.Net(models.densenet121(pretrained=False), 8)
        ckpt = torch.load("../checkpoints/densenet-121/checkpoint.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model.cuda()
    elif args.net_type == 'mobilenet':
        model = mobilenet.Net(models.mobilenet_v2(pretrained=False), 8)
        ckpt = torch.load("../checkpoints/mobilenet/checkpoint.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model.cuda()
        print("Done!")
    elif args.net_type =='resnet_50':
        model = resnet_50.Net(models.resnet50(pretrained=False), 8)
        ckpt = torch.load("../checkpoints/resnet-50/checkpoint.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model.cuda()
        print("Done!")
    elif args.net_type =='vgg_16':
        model = vgg_16.Net(models.vgg16_bn(pretrained=False), 8)
        ckpt = torch.load("../checkpoints/vgg-16/checkpoint.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model.cuda()
        print("Done!")
    else:
        raise Exception(f"There is no net_type={args.net_type} available.")

    in_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    print('load model: ' + args.net_type)
    
    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)

    # measure the performance
    M_list = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    T_list = [1, 10, 100, 1000]
    base_line_list = []
    ODIN_best_tnr = [0, 0, 0]*2
    ODIN_best_results = [0 , 0, 0]*2
    ODIN_best_temperature = [-1, -1, -1]*2
    ODIN_best_magnitude = [-1, -1, -1]*2
    for T in T_list:
        for m in M_list:
            magnitude = m
            temperature = T
            lib_generation.get_posterior(model, args.net_type, test_loader, magnitude, temperature, args.outf, True)
            out_count = 0
            print('Temperature: ' + str(temperature) + ' / noise: ' + str(magnitude)) 
            for out_dist in out_dist_list:
                out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
                print('Out-distribution: ' + out_dist) 
                lib_generation.get_posterior(model, args.net_type, out_test_loader, magnitude, temperature, args.outf, False)
                if temperature == 1 and magnitude == 0:
                    test_results = callog.metric(args.outf, ['PoT'])
                    base_line_list.append(test_results)
                else:
                    val_results = callog.metric(args.outf, ['PoV'])
                    if ODIN_best_tnr[out_count] < val_results['PoV']['TNR']:
                        ODIN_best_tnr[out_count] = val_results['PoV']['TNR']
                        ODIN_best_results[out_count] = callog.metric(args.outf, ['PoT'])
                        ODIN_best_temperature[out_count] = temperature
                        ODIN_best_magnitude[out_count] = magnitude
                out_count += 1
    
    # print the results
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print('Baseline method: in_distribution: ' + args.dataset + '==========')
    count_out = 0
    for results in base_line_list:
        print('out_distribution: '+ out_dist_list[count_out])
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('')
        count_out += 1
        
    print('ODIN method: in_distribution: ' + args.dataset + '==========')
    count_out = 0
    for results in ODIN_best_results:
        print('out_distribution: '+ out_dist_list[count_out])
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['PoT']['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['PoT']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['PoT']['AUOUT']), end='')
        print('temperature: ' + str(ODIN_best_temperature[count_out]))
        print('magnitude: '+ str(ODIN_best_magnitude[count_out]))
        print('')
        count_out += 1

    
    
if __name__ == '__main__':
    main()