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
import models
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
    
    out_dist_list = ['skin_cli', 'skin_derm', 'corrupted', 'corrupted_70', 'imgnet', 'nct', 'final_test']

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
    elif args.net_type == 'resnet_50':
        model = resnet_50.Net(models.resnet50(pretrained=False), 8)
        ckpt = torch.load("../checkpoints/resnet-50/checkpoint.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model.cuda()
        print("Done!")
    elif args.net_type == 'vgg_16':
        model = vgg_16.Net(models.vgg16_bn(pretrained=False), 8)
        ckpt = torch.load("../checkpoints/vgg-16/checkpoint.pth")
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model.cuda()
        print("Done!")
    else:
        raise Exception(f"There is no net_type={args.net_type} available.")

    in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    print('load model: ' + args.net_type)
    
    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    
    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,224,224).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, args.num_classes, feature_list, train_loader)
    
    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]

    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, args.num_classes, args.outf, \
                                                        True, args.net_type, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            
        for out_dist in out_dist_list:
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
            print('Out-distribution: ' + out_dist) 
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader, args.num_classes, args.outf, \
                                                             False, args.net_type, sample_mean, precision, i, magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.dataset , out_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)
    
if __name__ == '__main__':
    main()
