# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity, NMI
from utils.serialization import load_checkpoint
import torch 
import ast
import numpy as np



def extract_recalls(data, data_root, width, net, checkpoint, dim, batch_size, nThreads, pool_feature, gallery_eq_query, model=None, epoch=0, org_feature=False, save_txt="", args=None):
    

    gallery_feature, gallery_labels, query_feature, query_labels = \
        Model2Feature(data=data, root=data_root, width=width, net=net, checkpoint=checkpoint,
                    dim=dim, batch_size=batch_size, nThreads=nThreads, pool_feature=pool_feature, model=model, org_feature=org_feature, args=args)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    if gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))
    
    print(torch.sum(gallery_feature), "test")

    recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=data, args=args, epoch=epoch)

    labels = [x.item() for x in gallery_labels]

    nmi = NMI(gallery_feature, gallery_labels, n_cluster=len(set(labels)))
    print(recall_ks, nmi)
    result = '  '.join(['%.4f' % k for k in (recall_ks.tolist() + [nmi])])

    print('Epoch-%d' % epoch, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Testing')

    parser.add_argument('--data', type=str, default='cub')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--save_txt', type=str, default="")
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=False,
                        help='Is gallery identical with query')
    parser.add_argument('--net', type=str, default='VGG16-BN')
    parser.add_argument('--resume', '-r', type=str, default='model.pkl', metavar='PATH')

    parser.add_argument('--dim', '-d', type=int, default=512,
                        help='Dimension of Embedding Feather')
    parser.add_argument('--width', type=int, default=224,
                        help='width of input image')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                        help='if True extract feature from the last pool layer')
    
    args = parser.parse_args()

    org_feature = False
    print(args, org_feature)
    checkpoint = load_checkpoint(args.resume)
    epoch = checkpoint['epoch']
    extract_recalls(data=args.data, data_root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint,
                    dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature,
                    gallery_eq_query=args.gallery_eq_query, epoch=epoch, save_txt=args.save_txt, args=args, org_feature=org_feature)



