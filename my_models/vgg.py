# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

"""

import torch
from torch import nn


class Net (nn.Module):

    def __init__(self, vgg, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):

        super(Net, self).__init__()

        self.features = nn.Sequential(*list(vgg.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if feat_reducer is None:
            self.feat_reducer = nn.Sequential(
                nn.Linear(25088, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=p_dropout),
                nn.Linear(1024, neurons_class),
                nn.BatchNorm1d(neurons_class),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.feat_reducer = feat_reducer

        # Here comes the extra information (if applicable)
        if classifier is None:
            self.classifier = nn.Linear(neurons_class + n_extra_info, num_class)
        else:
            self.classifier = classifier

    def forward(self, img, extra_info=None):

        x = self.features(img)

        # Flatting
        x = x.view(x.size(0), -1)

        x = self.feat_reducer(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        res = self.classifier(agg)

        return res

