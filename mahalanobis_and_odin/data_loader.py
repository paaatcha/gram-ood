# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getSkinCancer(batch_size, dataroot):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sk_train = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(dataroot, 'skin_cancer', 'train'), transform=trans),
        batch_size=batch_size,
        shuffle=False)

    sk_test = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(dataroot, 'skin_cancer', 'test'), transform=trans),
        batch_size=batch_size,
        shuffle=False)

    return sk_train, sk_test


def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'skin_cancer':
        train_loader, test_loader = getISIC(batch_size, dataroot)
    else:
        raise Exception(f"There is no data_type={data_type} available!")

    return train_loader, test_loader

def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_c':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN'))
        testsetout = datasets.ImageFolder(dataroot, transform=transforms.Compose([transforms.CenterCrop(32),input_TF]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'imagenet_c':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet'))
        testsetout = datasets.ImageFolder(dataroot, transform=transforms.Compose([transforms.CenterCrop(32),input_TF]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'skin_cli':
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(dataroot, 'skin_cli'), transform=trans),
            batch_size=batch_size,
            shuffle=False)
    elif data_type == 'skin_derm':
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(dataroot, 'skin_derm'), transform=trans),
            batch_size=batch_size,
            shuffle=False)
    elif data_type == 'corrupted':
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(dataroot, 'corrupted'), transform=trans),
            batch_size=batch_size,
            shuffle=False)
    elif data_type == 'corrupted_70':
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(dataroot, 'corrupted_70'), transform=trans),
            batch_size=batch_size,
            shuffle=False)
    elif data_type == 'imgnet':
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(dataroot, 'imgnet'), transform=trans),
            batch_size=batch_size,
            shuffle=False)
    elif data_type == 'nct':
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(dataroot, 'nct'), transform=trans),
            batch_size=batch_size,
            shuffle=False)
    else:
        raise Exception(f"There is no data_type={data_type} available!")

    return test_loader


