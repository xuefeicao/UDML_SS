import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from models.BN_Inception import Embedding

class ResNet(nn.Module):
    """ ResNet-based image encoder to turn an image into a feature vector """

    def __init__(self, which_resnet="18", dim=512, self_supervision_rot=0, pretrained=False):
        super().__init__()
        if which_resnet == "18":
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif which_resnet == "34":
            resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif which_resnet == "50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif which_resnet == "101":
            resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif which_resnet == "152":
            resnet = torchvision.models.resnet152(pretrained=pretrained)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.dim = dim
        self.self_supervision_rot = self_supervision_rot
        if which_resnet in ["50", "101", "152"]:
            n_l = 2048
        else:
            n_l = 512
        self.classifier = Embedding(n_l, self.dim, normalized=True)
        all_extra_modules = list(self.classifier.modules())
        
        if self.self_supervision_rot:
            if which_resnet == "18":
                self.classifier_rot = nn.Sequential(
                nn.Linear(n_l, 512),
                nn.ReLU(True),
                nn.Linear(512, 4),
                #nn.ReLU(True),
                )
            else:
                self.classifier_rot = nn.Sequential(
                nn.Linear(n_l, 512),
                nn.ReLU(True),
                nn.Linear(512, 4),
                nn.ReLU(True),
                )
            print(self.classifier_rot)
            all_extra_modules += list(self.classifier_rot.modules())
        
        for m in all_extra_modules:
            if isinstance(m, nn.Conv2d):
                #print(m)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                #print(m)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                #print(m)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rot=False, org_feature=False):
        # b, g, r = x[:,0,:,:], x[:,1,:,:], x[:,2,:,:]
        # n, h, w = b.size()
        # b = b.view((n, 1, h, w))
        # g = g.view((n, 1, h, w))
        # r = r.view((n, 1, h, w))
        # x = torch.cat([r, g, b], dim=1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        if org_feature:
            return x
        if (not self.self_supervision_rot) or (not rot):
            y = self.classifier(x)
            return y 
        else:
            z = self.classifier_rot(x)
            return z


def resnet_all(which_resnet="18", dim=512, pretrained=True, self_supervision_rot=0):
    model = ResNet(which_resnet, dim, self_supervision_rot, pretrained=pretrained)
    return model
