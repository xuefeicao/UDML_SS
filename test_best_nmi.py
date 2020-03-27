# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity, NMI
from utils.serialization import load_checkpoint
import torch 
import ast
import os

def get_best(file):
    with open(file) as f:
        lines = f.readlines()
        lines = [line for line in lines if "Epoch" in line]
        dic_lines = dict()
        results = []
        for line in lines:
            line = line[6:].split()
            results.append([line[0], float(line[1])])
        print(results)
        return sorted(results, key=lambda x: x[1], reverse=True)[0][0]

def extract_recalls(data, data_root, width, net, checkpoint, dim, batch_size, nThreads, pool_feature, gallery_eq_query, epoch=0, model=None, org_feature=False, save_txt="", args=None):
    if args.data == "cifar":
        return
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
    parser.add_argument('--save_dir', '-r', type=str)

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
    if "rot_only" in args.save_txt or (args.dim % 64 != 0):
        org_feature = True
    else:
        org_feature = False
    print(args)
    epoch = get_best(args.save_txt)
    checkpoint = load_checkpoint(os.path.join(args.save_dir, "ckp_ep"+epoch+".pth.tar"))
    epoch = checkpoint['epoch']
    print(epoch)
    extract_recalls(data=args.data, data_root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint,
                    dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature,
                    gallery_eq_query=args.gallery_eq_query, epoch=epoch, save_txt=args.save_txt, args=args, org_feature=org_feature)



