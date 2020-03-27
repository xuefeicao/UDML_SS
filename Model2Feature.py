# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
cudnn.benchmark = True


def Model2Feature(data, net, checkpoint, dim=512, width=224, root=None, nThreads=16, batch_size=100, pool_feature=False, model=None, org_feature=False, args=None):
    dataset_name = data
    if model is None:
        model = models.create(net, dim=dim, pretrained=False)
        # resume = load_checkpoint(ckp_path)
        resume = checkpoint
        model.load_state_dict(resume['state_dict'], strict=False)
        model = torch.nn.DataParallel(model).cuda()
    data = DataSet.create(data, width=width, root=root, mode="test", self_supervision_rot=0, args=args)
    
    
    if dataset_name in ['shop', 'jd_test', 'cifar']:
        gallery_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=nThreads)

        query_loader = torch.utils.data.DataLoader(
            data.query, batch_size=batch_size,
            shuffle=False, drop_last=False,
            pin_memory=True, num_workers=nThreads)

        gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature, org_feature=org_feature)
        query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature, org_feature=org_feature)
        if org_feature:
            norm = query_feature.norm(dim=1, p=2, keepdim=True)
            query_feature = query_feature.div(norm.expand_as(query_feature))
            print("feature normalized 1")
            norm = gallery_feature.norm(dim=1, p=2, keepdim=True)
            gallery_feature = gallery_feature.div(norm.expand_as(gallery_feature))
            print("feature normalized 2")
    else:
        data_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size,
            shuffle=False, drop_last=False, pin_memory=True,
            num_workers=nThreads)
        features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature, org_feature=org_feature)
        if org_feature:
            norm = features.norm(dim=1, p=2, keepdim=True)
            features = features.div(norm.expand_as(features))
            print("feature normalized")
        gallery_feature, gallery_labels = query_feature, query_labels = features, labels
    return gallery_feature, gallery_labels, query_feature, query_labels

