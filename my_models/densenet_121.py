# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import torch
from torch import nn
import torch.nn.functional as nnF


class Net (nn.Module):

    def __init__(self, densenet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):
        
        super(Net, self).__init__()
        
        self.features = nn.Sequential(*list(densenet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        if feat_reducer is None:
            self.feat_reducer = nn.Sequential(
                nn.Linear(1024, neurons_class),
                nn.BatchNorm1d(neurons_class),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.feat_reducer = feat_reducer

        if classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(neurons_class + n_extra_info, num_class)
            )
        else:
            self.classifier = classifier


    def forward(self, img, extra_info=None):

        xf = self.features(img)
        x = nnF.relu(xf, inplace=True)
        x = nnF.adaptive_avg_pool2d(x, (1, 1)).view(xf.size(0), -1)

        x = self.feat_reducer(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        x = self.classifier(agg)

        return x 


