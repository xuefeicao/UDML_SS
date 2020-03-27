# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity, NMI
from utils.serialization import load_checkpoint
import torch 
import ast
import numpy as np
def knn_clas(sim_mat, gallery_labels, query_labels, k=200, sigma=0.1, args=None):
    sim_mat = sim_mat.numpy()
    gallery_labels = [gallery_labels[i].item() for i in range(len(gallery_labels))]
    query_labels = [query_labels[i].item() for i in range(len(query_labels))]
    if hasattr(args, "save_txt"):
        save_npy = args.save_txt.replace(".txt", "-") + str(epoch) + "_sim.npy"
        np.save(save_npy, sim_mat)

    ans = []
    for i in range(len(query_labels)):
        yi = np.argsort(sim_mat[i,:])[-k:]
        
        dic = {}
        for ind in yi:
            if gallery_labels[ind] not in dic:
                dic[gallery_labels[ind]] = 0
            dic[gallery_labels[ind]] += np.exp(sim_mat[i, ind]/sigma)
        
        allkeys = list(dic.keys())
        prelabel = sorted(allkeys, key=lambda x: -dic[x])[0]
        if i == 0:
            print(dic, prelabel, k, sigma)
        ans.append(prelabel)
    return np.mean([query_labels[i]==ans[i] for i in range(len(query_labels))])
        







def extract_recalls(data, data_root, width, net, checkpoint, dim, batch_size, nThreads, pool_feature, gallery_eq_query, model=None, epoch=0, org_feature=False, save_txt="", args=None):
    return_early = False
    if save_txt != "":
        with open(save_txt) as f:
            allfile = f.read()
        with open(save_txt) as f:
            results = f.readlines()
            results = [result for result in results if "Epoch-" in result]
        if len(results) > 0:
            tmp = [int(result.split()[0][6:]) for result in results]
            old_epoch = max(tmp)
            print(old_epoch, epoch, "ep")
            if (old_epoch >= epoch):
                return_early = True
                return

    gallery_feature, gallery_labels, query_feature, query_labels = \
        Model2Feature(data=data, root=data_root, width=width, net=net, checkpoint=checkpoint,
                    dim=dim, batch_size=batch_size, nThreads=nThreads, pool_feature=pool_feature, model=model, org_feature=org_feature, args=args)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    if gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))
    else:
        ans = []
        for k in [10, 100]:
            for sigma in [0.01, 0.1, 1]:
                ans.append(knn_clas(sim_mat, gallery_labels, query_labels, k, sigma, args))
        result = " ".join([str(x) for x in ans])
        print('Epoch-%d' % epoch, result)
        return 
    print(torch.sum(gallery_feature), "test")

    recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=data, args=args, epoch=epoch)
    if return_early:
        print("**")
        return
    labels = [x.item() for x in gallery_labels]
    if (args.data == "product"):
        nmi = 0
    else:
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
    if "rot_only" in args.save_txt or (args.dim % 64 != 0):
        org_feature = True
    else:
        org_feature = False
    print(args, org_feature)
    checkpoint = load_checkpoint(args.resume)
    epoch = checkpoint['epoch']
    extract_recalls(data=args.data, data_root=args.data_root, width=args.width, net=args.net, checkpoint=checkpoint,
                    dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature,
                    gallery_eq_query=args.gallery_eq_query, epoch=epoch, save_txt=args.save_txt, args=args, org_feature=org_feature)



